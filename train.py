import numpy as np
import logging.config
from argparse import ArgumentParser, Namespace
import json
import time
from datetime import timedelta, datetime


import tensorflow as tf
import contextlib


from utils.common import (load_dataset,
                        sample_k_fids_for_pid,
                        fid_to_image)

from utils.dirs import directories
from utils import os_utils
import utils.lbtoolbox as lb
from signal import SIGINT, SIGTERM
import time
from datetime import timedelta, datetime

from model.embedding_model import  EmbeddingModel



from ranking import LOSS_CHOICES,METRIC_CHOICES
from ranking.npair import npairs_loss
from ranking.angular import angular_loss
from ranking.hard_triplet import batch_hard
from ranking.contrastive import contrastive_loss
from ranking.lifted_structured import lifted_loss
from ranking.semi_hard_triplet import triplet_semihard_loss



import os


parser = ArgumentParser(description='Entrenar sistemas CBIR')

parser.add_argument(
    '-c', '--config', required=True, type=str,
    help='Fichero de configuración del modelo')


def main(cfg):

    #Configuramos para utitilizar toda la memoria de los dispoisitvo GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:

                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


    # TODO:  Habilitar loggers
    # Definimos la ubicación donde se almacena los logs de Tensorboard
    train_summary_writer = tf.summary.create_file_writer(cfg.dirs.train_log)


    # Preparamos el DATASET
    # Cargamos los datos desde el fichero CSV
    pids, fids = load_dataset(cfg.dirs.csv_file, cfg.dirs.images)

    # Obtenemos todas las etiquetas
    unique_pids = np.unique(pids)
    print ("Etiquetas únicas: {}".format(len(unique_pids)))
    # Preparamos un dataset donde en cada época se  se cubran todos los valores de PID
    # y si se distribuyan uniformemente.
    dataset = tf.data.Dataset.from_tensor_slices(unique_pids)
    if len(unique_pids) < cfg.model.batch.P:
        unique_pids = np.tile(unique_pids, int(np.ceil(cfg.model.batch.P / len(unique_pids))))
    dataset = tf.data.Dataset.from_tensor_slices(unique_pids)
    # Cogemos valores aleatorios
    dataset = dataset.shuffle(len(unique_pids))
    # Forzamos que el tamaño del dataset sea múltiplo de batch-size
    dataset = dataset.take((len(unique_pids) // cfg.model.batch.P) * cfg.model.batch.P)
    dataset = dataset.repeat(None)

    # Para cada PID obtenemos el cfg.model.batch.K imagenes
    dataset = dataset.map(lambda pid: sample_k_fids_for_pid(
        pid, all_fids=fids, all_pids=pids, batch_k=cfg.model.batch.K))

    # Desagrupamos los cfg.model.batch.K para una mejor carga de las imágenes
    dataset = dataset.unbatch()

    # Ahora cargamos las imágenes transformandolas al tamaño del emb_modelo

    net_input_size = (cfg.model.input.height, cfg.model.input.width)
    pre_crop_size = (cfg.model.crop.height, cfg.model.crop.width)

    # Comprobamos si queremos hacer el crop

    image_size = pre_crop_size if cfg.model.crop else net_input_size

    # Redimensionamos las imágenes
    dataset = dataset.map(lambda fid, pid: fid_to_image (
        fid, pid, cfg.dirs.images, image_size), num_parallel_calls=cfg.loading_threads)

    # Si se ha habilitado el CROP redimensionamos al tamaño de red

    if cfg.model.crop:
        dataset = dataset.map(lambda im, fid, pid:
                              (tf.image.random_crop(im, net_input_size + (3,)),
                               fid,
                               pid))



    #Carga de y un model Predefinido y preparación para el entrenamiento
    # Definimos la fase de nuestro entorno TF 0 = test, 1= train
    tf.keras.backend.set_learning_phase(1)
    emb_model = EmbeddingModel(cfg)
    # Agrupamos el dataset en los batch_size
    # y preprocesamos la imagen para prepararlo para el emb_modelo
    batch_size = cfg.model.batch.P * cfg.model.batch.K
    dataset = dataset.map(lambda im, fid, pid: (emb_model.preprocess_input(im), fid, pid))
    dataset = dataset.batch(batch_size)

    print ('Batch-size: {}'.format(batch_size))

    # Preparación de los siguientes batch
    # Esto mejora la latencia y el rendimiento en el coste computacional de usar
    # memoria adicional  para almacenar los siguientes batch
    dataset = dataset.prefetch(2)


    # Establecemos el optimizador y la programación ratio de aprendizaje (learning-rate schedule)
    if 0 <=cfg.model.fit.decay_start_iteration  <cfg.model.fit.epochs:
           cfg.model.fit.lr = tf.optimizers.schedules.PolynomialDecay(cfg.model.fit.lr,cfg.model.fit.epochs,
                                                      end_learning_rate=1e-7)
    else:
       cfg.model.fit.lr =cfg.model.fit.lr


    if cfg.model.fit.optimizer== 'adam':
       optimizer= tf.keras.optimizers.Adam(cfg.model.fit.lr)
    elif cfg.model.fit.optimizer== 'SGD':
       optimizer= tf.keras.optimizers.SGD(cfg.model.fit.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizador no válido {}'.format(cfg.model.fit.optimizer))


    # Iniciamos el entrenamiento

    start_step = 0
    dataset_iter = iter(dataset)

    # Definimos los checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=emb_model)
    manager = tf.train.CheckpointManager(ckpt, cfg.dirs.checkpoint, max_to_keep=3)

    # Recuperamos el último checkpoint
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Recuperado desde {}".format(manager.latest_checkpoint))
    else:
        print("Inicializado desde inicio.")

    # Almacenamos los datos para tensorboard
    tf.summary.trace_on(graph=True, profiler=True)

    #  Función para facilmente cambiar los estados del optimizador
    @contextlib.contextmanager
    def options(options):
      old_opts = tf.config.optimizer.get_experimental_options()
      tf.config.optimizer.set_experimental_options(options)
      try:
        yield
      finally:
        tf.config.optimizer.set_experimental_options(old_opts)




    @tf.function(experimental_relax_shapes=True)
    def train_step(images, pids, iteration ):
        cfg = emb_model.cfg
        with tf.GradientTape() as tape:
            # Obtenemos de cada batch los correspondientes vectores
            # de característias
            batch_embedding = emb_model(images )

            # Realizamos la norma de orden 2 si procede
            if emb_model.l2_embedding:
                batch_embedding = tf.nn.l2_normalize(batch_embedding, -1)
            else:
                batch_embedding = batch_embedding
            # Aplicacmos una función de perdia
            if cfg.model.fit.loss == 'semi_hard_triplet':
                embedding_loss = triplet_semihard_loss(batch_embedding, pids, cfg.model.fit.margin)
            elif cfg.model.fit.loss == 'hard_triplet':
                embedding_loss = batch_hard(batch_embedding, pids, cfg.model.fit.margin, cfg.model.fit.metric)
            elif cfg.model.fit.loss == 'lifted_loss':
                embedding_loss = lifted_loss(pids, batch_embedding, margin=cfg.model.fit.margin)
            elif cfg.model.fit.loss == 'contrastive_loss':
                assert batch_size % 2 == 0
                assert cfg.model.batch.K == 4  ## Can work with other number but will need tuning

                contrastive_idx = np.tile([0, 1, 4, 3, 2, 5, 6, 7], cfg.model.batch.P // 2)
                for i in range(cfg.model.batch.P // 2):
                    contrastive_idx[i * 8:i * 8 + 8] += i * 8

                contrastive_idx = np.expand_dims(contrastive_idx, 1)
                batch_embedding_ordered = tf.gather_nd(batch_embedding, contrastive_idx)
                pids_ordered = tf.gather_nd(pids, contrastive_idx)
                # batch_embedding_ordered = tf.Print(batch_embedding_ordered,[pids_ordered],'pids_ordered :: ',summarize=1000)
                embeddings_anchor, embeddings_positive = tf.unstack(
                    tf.reshape(batch_embedding_ordered, [-1, 2, cfg.model.embedding_dim]), 2,
                    1)
                # embeddings_anchor = tf.Print(embeddings_anchor,[pids_ordered,embeddings_anchor,embeddings_positive,batch_embedding,batch_embedding_ordered],"Tensors ", summarize=1000)

                fixed_labels = np.tile([1, 0, 0, 1], cfg.model.batch.P // 2)
                # fixed_labels = np.reshape(fixed_labels,(len(fixed_labels),1))
                # print(fixed_labels)
                labels = tf.constant(fixed_labels)
                # labels = tf.Print(labels,[labels],'labels ',summarize=1000)
                embedding_loss = contrastive_loss(labels, embeddings_anchor, embeddings_positive,
                                                margin=cfg.model.fit.margin)
            elif cfg.model.fit.loss == 'angular_loss':
                embeddings_anchor, embeddings_positive = tf.unstack(
                    tf.reshape(batch_embedding, [-1, 2, cfg.model.embedding_dim]), 2,
                    1)
                # pids = tf.Print(pids, [pids], 'pids:: ', summarize=100)
                pids, _ = tf.unstack(tf.reshape(pids, [-1, 2, 1]), 2, 1)
                # pids = tf.Print(pids,[pids],'pids:: ',summarize=100)
                embedding_loss = angular_loss(pids, embeddings_anchor, embeddings_positive,
                                            batch_size=cfg.model.batch.P, with_l2reg=True)

            elif cfg.model.fit.loss == 'npairs_loss':
                assert cfg.model.batch.K == 2  ## Single positive pair per class
                embeddings_anchor, embeddings_positive = tf.unstack(
                    tf.reshape(batch_embedding, [-1, 2, cfg.model.embedding_dim]), 2, 1)
                pids, _ = tf.unstack(tf.reshape(pids, [-1, 2, 1]), 2, 1)
                pids = tf.reshape(pids, [-1])
                embedding_loss = npairs_loss(pids, embeddings_anchor, embeddings_positive)

            else:
                raise NotImplementedError('Invalid Loss {}'.format(cfg.model.fit.loss))
            loss_mean = tf.reduce_mean(embedding_loss)


        gradients = tape.gradient(loss_mean, emb_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, emb_model.trainable_variables))
        # Almacenamos los datos para tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('Loss mean', loss_mean, step=iteration)
            # tf.summary.image("Training data", images, step=iteration)
            # tf.summary.scalar('Learning Rate', optimizer.lr, step=iteration)

        return embedding_loss

    #print('Starting training from iteration {}.'.format(start_step))
    with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:
            for i in range(ckpt.step.numpy(),cfg.model.fit.epochs):
                # for batch_idx, batch in enumerate():
                start_time = time.time()
                images, fids, pids = next(dataset_iter)
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    batch_loss = train_step(images, pids, i)
                elapsed_time = time.time() - start_time
                seconds_todo = (cfg.model.fit.epochs - i) * elapsed_time
                print('iter:{:6d}, loss min|avg|max: {:.3f}|{:.3f}|{:6.3f}, ETA: {} ({:.2f}s/it)'.format(
                    i,
                    tf.reduce_min(batch_loss).numpy(),tf.reduce_mean(batch_loss).numpy(),tf.reduce_max(batch_loss).numpy(),
                    # cfg.model.batch.K - 1, float(b_prec_at_k),
                    timedelta(seconds=int(seconds_todo)),
                    elapsed_time))

                ckpt.step.assign_add(1)
                if (cfg.checkpoint_frequency> 0 and i % cfg.checkpoint_frequency== 0):

                    #uncomment if you want to save the emb_model weight separately
                    #emb_model.save_weights(os.path.join(cfg.dirs.checkpoint, 'emb_model_weights_{0:04d}.w'.format(i)))
                    manager.save()

                # Stop the main-loop at the end of the step, if requested.
                if u.interrupted:
                    log.info("Interrupted on request!")
                    break


if __name__ == '__main__':

    args = parser.parse_args()
    config_file = args.config

    with open(config_file) as f:
        config = json.load(f, object_hook=lambda d: Namespace(**d))



    config = directories(config)

    main(config)

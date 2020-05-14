#!/usr/bin/env python3
import os
import h5py
import json
from utils import common
import configuration as const
from utils.dirs import directories
from itertools import count
import numpy as np
import tensorflow as tf
import time

# from sklearn.cluster import KMeans
# from sklearn.metrics import normalized_mutual_info_score

from argparse import ArgumentParser, Namespace

from utils import os_utils
from utils.common import *
from utils.metrics import *

from model.embedding_model import  EmbeddingModel



parser = ArgumentParser(description='Crear la base de datos de vectores de características desde una red entrenada')
# Required
parser.add_argument(
    '-c', '--config', required=True, type=str,
    help='Fichero de configuración del modelo')

parser.add_argument(
    '-b', '--batch-size', required=False, type=int, default=64,
    help='Tamaño de los batch para crear la base de datos de los vectores de característias')

parser.add_argument(
    '-r', '--n_retrievals', required=False, type=int, default=10,
    help='Número de imágenes recuperadas')

parser.add_argument(
    '-t', '--n_test_samples', required=False, type=int, default=100,
    help='Número de imágenes para hacer el cálculo del MAP')




def main(cfg):

    # TODO:  Habilitar loggers
    # Definimos la ubicación donde se almacena los logs de Tensorboard
    eval_summary_writer = tf.summary.create_file_writer(cfg.dirs.eval_log)

    if not os.path.exists(cfg.dirs.csv_file):
        raise  IOError(' No se encuentra el fichero del dataset: {}'.format(cfg.dirs.csv_file))
        return


    # Carga de los datos de nuestro catálogo

    # Cargamos el fichero con las característias y los pids de nuestro
    # catálogo (base de datos de imágenes)

    with h5py.File(cfg.dirs.embeddings_file, 'r') as db:
            db_embs = db['emb'][()]
            db_pids = db['pids'][()]
            db_fids = db['fids'][()]

    # Asignamos un idx a cada imagen

    idxs=np.array([i for i in np.arange(len(db_fids))])
    # Creamos el dataset con nuestro catálogo
    dataset = tf.data.Dataset.from_tensor_slices((db_embs, db_pids, idxs))



    # Cargamos los datos desde el fichero de Test
    test_pids, test_fids = load_dataset(cfg.dirs.test_file, cfg.dirs.images)
    # Preparamos un dataset para pasar el modelo y obtener los
    # vectores de características de toda nuestras imaǵenes de test
    test_dataset = tf.data.Dataset.from_tensor_slices((test_fids, np.asarray(test_pids, dtype=int)))
    # Cogemos valores aleatorios
    test_dataset = test_dataset.shuffle(len(test_fids))
    # Ahora cargamos las imágenes transformandolas al tamaño del emb_modelo
    net_input_size = (cfg.model.input.height, cfg.model.input.width)
    pre_crop_size = (cfg.model.crop.height, cfg.model.crop.width)
    # Comprobamos si queremos hacer el crop
    image_size = pre_crop_size if cfg.model.crop else net_input_size
    # Redimensionamos las imágenes
    test_dataset = test_dataset.map(lambda fid, pid: fid_to_image (
        fid, pid, cfg.dirs.images, image_size), num_parallel_calls=cfg.loading_threads)
    # Si se ha habilitado el CROP redimensionamos al tamaño de red
    if cfg.model.crop:
        test_dataset = test_dataset.map(lambda im, fid, pid:
                              (tf.image.random_crop(im, net_input_size + (3,)),
                               fid,
                               pid))
    #Carga de y un model Predefinido y preparación para el entrenamiento
    # Definimos la fase de nuestro entorno TF 0 = test, 1= train
    tf.keras.backend.set_learning_phase(0)
    emb_model = EmbeddingModel(cfg)
    # Agrupamos el dataset en los batch_size
    # y preprocesamos la imagen para prepararlo para el emb_modelo
    test_dataset = test_dataset.map(lambda im, fid, pid: (emb_model.preprocess_input(im), fid, pid, im))

    # Para cada un de los test tenemos que coger una imágen de test
    # que será nuestra imagen Query
    # TODO: Podemos ver otra opción de coger el número de purebas
    test_dataset = test_dataset.batch(1)
    test_dataset_iter = iter(test_dataset)

    # Preparación de los siguientes batch
    # Esto mejora la latencia y el rendimiento en el coste computacional de usar
    # memoria adicional  para almacenar los siguientes batch
    test_dataset = test_dataset.prefetch(1)


    # Definimos los checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=emb_model)
    manager = tf.train.CheckpointManager(ckpt, cfg.dirs.checkpoint, max_to_keep=1)
    # Recuperamos el último checkpoint
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Recuperado desde {}".format(manager.latest_checkpoint))
    else:
        print("Inicializado desde inicio.")

    # Para cada una de las imagenes de respuesta grabamos la etiqueta y las distancias.
    matched_labels = []
    distances = []
    retrieved_idxs = []
    # Recorremos todas imagénes del db_queries y obtenemos las imágenes más cercanas
    total_t0 = time.time()
    # Repetimos este proceso por n_test_samples
    maps= []
    for i in range(cfg.n_test_samples):
        t0 = time.time()
        # obtenemos el vector de nuestra imagen
        test_image, test_fid, test_pid , test_original = next(test_dataset_iter)
        query= emb_model(test_image)
        #Calculamos las distancias
        dataset_distances = dataset.map( lambda embs, pid, idx: calculate_distances(embs, pid,idx, query,type='Euclidian'))
        # Quitamos las caraterísitas
        dataset_without_embs = dataset_distances.map( lambda embs, pid, idx, distance: remove_embs(embs, pid,idx, distance))
        dataset_without_embs = dataset_without_embs.map( lambda  pid, idx, distance:  boolean_label( pid,idx, distance, test_pid))
        # TODO: Ver una forma de ordenar con Tensorflow
        # el array se quedaría pid,idx, distance, boolean_label
        distance_with_labels = np.array(list(dataset_without_embs.as_numpy_iterator()))
        sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 2].argsort()]
        # Obtenemos los indices, distancia y etiquetas
        # sólo del número que vamos a devolver (n_retrievals)
        sorted_idxs = np.array(sorted_distance_with_labels[:n_retrievals,1]).astype('int')
        sorted_labels = np.array(sorted_distance_with_labels[:n_retrievals, 3]).astype('int')
        sorted_distances = np.array(sorted_distance_with_labels[:n_retrievals, 2]).astype('float32')

        # Calculamos nuesto AP@ke
        # k es el número de imageners recuperas (n_retrievals)
        print("Label query: {}".format(test_pid))
        print("Labels images retrieval: {}". format(sorted_distance_with_labels[:n_retrievals,0]))

        ap, aps = APatk(sorted_labels)
        maps.append(ap)

        # retrieved_idxs.append(sorted_idxs)
        # distances.append(sorted_distances)
        # matched_labels.append(sorted_labels)


        # Obtenemos las imágenes recuperadas
        retrievals = tf.data.Dataset.from_tensor_slices((db_fids[sorted_idxs],
                                                        np.asarray(db_pids, dtype=int)[sorted_idxs]))
        # Redimensionamos las imágenes
        retrievals  = retrievals .map(lambda fid, pid: fid_to_image (
            fid, pid, cfg.dirs.images, image_size), num_parallel_calls=cfg.loading_threads)

        # Si se ha habilitado el CROP redimensionamos al tamaño de red

        # if cfg.model.crop:
        #     retrievals  = retrievals .map(lambda im, fid, pid:
        #                           (tf.image.random_crop(im, net_input_size + (3,)),
        #                            fid,
        #                            pid))
        #Cogemos una imágen para hacer la prueba
        retrievals = retrievals.batch(cfg.n_retrievals)
        retrievals_iter = iter(retrievals)
        retrievals_images, retrievals_fid, retrievals_pid = next(retrievals_iter)


        summary_retrievals=show_images(test_original, test_fid, test_pid,
                                       retrievals_images, retrievals_fid, retrievals_pid, sorted_distances,
                                       columns=4,
                                       ap=ap, aps=aps
                                       )
        t1 = time.time()
        with eval_summary_writer.as_default():
            tf.summary.image("Imagenes Recuperadas", plot_to_image(summary_retrievals), step=i)
            tf.summary.scalar("ap", ap, step=i)
            tf.summary.scalar('Tiempo de recuperación', t1-t0, step=i)


    total_t1 = time.time()
    print('Tiempo en recuperar las imágenes: %.4f s' % (t1-t0))
    # output=np.stack((distances, matched_labels, retrieved_idxs), axis=-1)
    # score = label_ranking_average_precision_score(matched_labels, distances)
    # print('Model score: %.2f %%' % (score*100))
    map = np.average(np.array(maps))
    print('MAP: {:.2f}%'.format(map*100))
    with eval_summary_writer.as_default():
        tf.summary.scalar('Tiempo Total de recuperación', total_t1-total_t0, step=1)
        tf.summary.scalar('Score mAP', map, step=1)







if __name__ == '__main__':

    args = parser.parse_args()

    config_file = args.config
    batch_size = args.batch_size
    n_retrievals = args.n_retrievals
    n_test_samples = args.n_test_samples

    with open(config_file) as f:
        config = json.load(f, object_hook=lambda d: Namespace(**d))

    config = directories(config, state='test')
    config.batch_size = batch_size

    config.n_retrievals = n_retrievals
    config.n_test_samples = n_test_samples

    main(config)

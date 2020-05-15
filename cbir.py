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


parser = ArgumentParser(description='Ejemplo de recuperación de imágenes desde una imagen de Ejemplo')
# Required

parser.add_argument(
    'image', type=str,
    help='Imagen sobre la que devolver imágenes similares')

parser.add_argument(
    '-c', '--config', required=True, type=str,
    help='Fichero de configuración del modelo')

parser.add_argument(
    '-r', '--n_retrievals', required=False, type=int, default=10,
    help='Número de imágenes recuperadas')




def main(cfg):

    # TODO:  Habilitar loggers
    # Definimos la ubicación donde se almacena los logs de Tensorboard
    cbir_summary_writer = tf.summary.create_file_writer(cfg.dirs.cbir_log)

    if not os.path.exists(cfg.dirs.csv_file):
        raise  IOError(' No se encuentra el fichero del dataset: {}'.format(cfg.dirs.csv_file))
        return

    # Cargamos los nombres de la etiquetas

    labels_id, labels_name = load_labels(cfg.dirs.labels_file)
    # print(labels_id)
    # print(labels_name)
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





    # Ahora cargamos las imágenes transformandolas al tamaño del emb_modelo
    net_input_size = (cfg.model.input.height, cfg.model.input.width)
    pre_crop_size = (cfg.model.crop.height, cfg.model.crop.width)
    # Comprobamos si queremos hacer el crop
    image_size = pre_crop_size if cfg.model.crop else net_input_size
    # Redimensionamos las imágenes
    query_image = load_image(query_file, image_size)

    # Si se ha habilitado el CROP redimensionamos al tamaño de red
    if cfg.model.crop:
        query_image =tf.image.random_crop(query_image, net_input_size + (3,))
    #Carga de y un model Predefinido y preparación para el entrenamiento
    # Definimos la fase de nuestro entorno TF 0 = test, 1= train
    tf.keras.backend.set_learning_phase(0)
    emb_model = EmbeddingModel(cfg)
    # Agrupamos el dataset en los batch_size
    # y preprocesamos la imagen para prepararlo para el emb_modelo
    query_image_preprocessed = emb_model.preprocess_input(query_image)

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
    t0 = time.time()
    # Obtenemos el vector de nuestra imagen
    query_image_preprocessed = tf.reshape(query_image_preprocessed, [1, cfg.model.input.height, cfg.model.input.width, 3])

    query= emb_model(query_image_preprocessed)

    #Calculamos las distancias
    dataset = dataset.map( lambda embs, pid, idx: calculate_distances(embs, pid,idx, query,type='Euclidian'))
    # Quitamos las caraterísitas para agilizar los cálculos
    dataset_without_embs = dataset.map( lambda embs, pid, idx, distance: remove_embs(embs, pid,idx, distance))

    # TODO: Ver una forma de ordenar con Tensorflow GPU
    # el array se quedaría pid,idx, distance, boolean_label
    distance_with_labels = np.array(list(dataset_without_embs.as_numpy_iterator()))
    sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 2].argsort()]

    # Obtenemos los indices, distancia y etiquetas
    # sólo del número que vamos a devolver (n_retrievals)
    sorted_idxs = np.array(sorted_distance_with_labels[:n_retrievals,1]).astype('int')
    sorted_distances = np.array(sorted_distance_with_labels[:n_retrievals, 2]).astype('float32')
    print (sorted_distances)


    # Obtenemos los nombres de las etiquetas de las imágenes obtenidas
    sorted_labels = sorted_distance_with_labels[:n_retrievals, 0]
    sorted_labels_names =  [get_label_name(pid,labels_id, labels_name ) for pid in sorted_labels]
    print("Labels images retrieval: {}". format(sorted_labels_names))

    retrievals = tf.data.Dataset.from_tensor_slices((db_fids[sorted_idxs],
                                                    np.asarray(db_pids, dtype=int)[sorted_idxs]))
    # Redimensionamos las imágenes
    retrievals  = retrievals .map(lambda fid, pid: fid_to_image (
        fid, pid, cfg.dirs.images, image_size), num_parallel_calls=cfg.loading_threads)
    # print(list(retrievals.as_numpy_iterator()))
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

    query_image =tf.reshape(query_image, [1, cfg.model.input.height, cfg.model.input.width, 3])
    summary_retrievals=show_images(query_image, [query_file], [-1],
                                   retrievals_images, retrievals_fid, retrievals_pid, sorted_distances)

    print (retrievals_fid)
    print (retrievals_pid)
    t1 = time.time()
    print('Tiempo en recuperar las imágenes: %.4f s' % (t1-t0))

    with cbir_summary_writer.as_default():
        tf.summary.image("Imagenes Recuperadas", plot_to_image(summary_retrievals), step=1)
        tf.summary.scalar('Tiempo de recuperación', t1-t0, step=1)






if __name__ == '__main__':

    args = parser.parse_args()
    config_file = args.config
    n_retrievals = args.n_retrievals
    query_file = args.image
    with open(config_file) as f:
        config = json.load(f, object_hook=lambda d: Namespace(**d))

    config = directories(config, state='production')
    config.n_retrievals = n_retrievals


    main(config)

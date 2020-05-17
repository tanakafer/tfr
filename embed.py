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
from tensorboard.plugins import projector

from argparse import ArgumentParser, Namespace

from utils import os_utils
from utils.common import *

from PIL import Image
from model.embedding_model import  EmbeddingModel

parser = ArgumentParser(description='Crear la base de datos de vectores de características desde una red entrenada')
# Required
parser.add_argument(
    '-c', '--config', required=True, type=str,
    help='Fichero de configuración del modelo')

parser.add_argument(
    '-b', '--batch-size', required=False, type=int, default=64,
    help='Tamaño de los batch para crear la base de datos de los vectores de característias')






def main(cfg):




    if not os.path.exists(cfg.dirs.csv_file):
        raise  IOError(' No se encuentra el fichero del dataset: {}'.format(cfg.dirs.csv_file))
        return
    # Preparamos el DATASET
    # Cargamos los datos desde el fichero CSV
    pids, fids = load_dataset(cfg.dirs.csv_file, cfg.dirs.images)


    # Preparamos un dataset para pasar el modelo y obtener los
    # vectores de características de toda nuestra base de datos
    dataset = tf.data.Dataset.from_tensor_slices(fids)
    # Asignamos a cada imagen su pid
    dataset = dataset.map(lambda fid: find_pid(fid, fids, pids))


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
    tf.keras.backend.set_learning_phase(0)
    emb_model = EmbeddingModel(cfg)

    # Agrupamos el dataset en los batch_size
    # y preprocesamos la imagen para prepararlo para el emb_modelo
    # Añadimos los thumbnails
    thumb_size=(28,28)
    dataset = dataset.map( lambda im, fid, pid: (im, fid, pid,tf.image.resize(im, thumb_size)))

    dataset = dataset.map(lambda im, fid, pid, thumb_size: (emb_model.preprocess_input(im), fid, pid, thumb_size))

    # dataset = dataset.map( lambda im, fid, pid, thumb: (im, fid, pid,tf.image.convert_image_dtype(thumb, dtype=tf.uint8, saturate=False)))

    dataset = dataset.batch(cfg.batch_size)
    print ('Batch-size: {}'.format(cfg.batch_size))

    # Preparación de los siguientes batch
    # Esto mejora la latencia y el rendimiento en el coste computacional de usar
    # memoria adicional  para almacenar los siguientes batch
    dataset = dataset.prefetch(2)


    # Augment the data if specified by the arguments.
    # `modifiers` is a list of strings that keeps track of which augmentations
    # have been applied, so that a human can understand it later on.
    modifiers = ['original']

    # TODO: Analizar si es necesario
    # if args.flip_augment:
    #     dataset = dataset.map(flip_augment)
    #     dataset = dataset.apply(tf.contrib.data.unbatch())
    #     modifiers = [o + m for m in ['', '_flip'] for o in modifiers]
    #
    # if args.crop_augment == 'center':
    #     dataset = dataset.map(lambda im, fid, pid:
    #         (five_crops(im, net_input_size)[0], fid, pid))
    #     modifiers = [o + '_center' for o in modifiers]
    # elif args.crop_augment == 'five':
    #     dataset = dataset.map(lambda im, fid, pid: (
    #         tf.stack(five_crops(im, net_input_size)),
    #         tf.stack([fid]*5),
    #         tf.stack([pid]*5)))
    #     dataset = dataset.apply(tf.contrib.data.unbatch())
    #     modifiers = [o + m for o in modifiers for m in [
    #         '_center', '_top_left', '_top_right', '_bottom_left', '_bottom_right']]
    # elif args.crop_augment == 'avgpool':
    #     modifiers = [o + '_avgpool' for o in modifiers]
    # else:
    #     modifiers = [o + '_resize' for o in modifiers]



    # Empezamos a sacar todas las característias de las imágenes del dataset
    with h5py.File(cfg.dirs.embeddings_file, 'w') as f_out:
        # Definimos los checkpoint
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=emb_model)
        manager = tf.train.CheckpointManager(ckpt, cfg.dirs.checkpoint, max_to_keep=1)
        # Recuperamos el último checkpoint
        ckpt.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            print("Recuperado desde {}".format(manager.latest_checkpoint))
        else:
            print("Inicializado desde inicio.")

        # Inicializamos la base de datos con ceros
        emb_storage = np.zeros(
            ((len(fids) * len(modifiers)), cfg.model.embedding_dim), np.float32)
        thumbnails = np.zeros(
            ((len(fids) * len(modifiers)), thumb_size[0], thumb_size[1], 3), np.float32)
        # Recorremos todo el dataset para ir almacenando
        # los vectores de características
        dataset_iter = iter(dataset)
        for start_idx in count(step=cfg.batch_size):
            try:
                images, _,_, thumbs = next(dataset_iter)
                emb = emb_model(images)
                emb_storage[start_idx:start_idx + len(emb)] += emb
                thumbnails[start_idx:start_idx + len(thumbs)] += thumbs
                print('\rCreando vectores de características {}-{}/{}'.format(
                    start_idx, start_idx + len(emb), len(emb_storage)),
                    flush=True, end='')
            except StopIteration:
                break  # This just indicates the end of the dataset
        # Almacenamos el vector con identificadores de las clases
        pids_dataset = f_out.create_dataset('pids', data=np.array(pids, np.int))
        # Almacenamos los nombres de las imágenes de las clases
        #Primero convertimos a ascii los nombres por un problem ane h5py
        #para gestionar los utf-8
        temp=[]
        for item in fids:
            temp.append(item.encode('ascii'))
        fids = np.array(temp)

        fids_dataset = f_out.create_dataset('fids', data=fids)
        # Almacenamos el vector de características
        emb_dataset = f_out.create_dataset('emb', data=emb_storage)
    f_out.close()

    sprite = create_sprite(thumbnails)



    sprite =  Image.fromarray(np.uint8(sprite))
    sprite_file =os.path.join(cfg.dirs.emb_log, "sprites.png")
    sprite.save(sprite_file)
    # Almacenamos el checkpoint para el projector



    emb_storage = tf.Variable(emb_storage, name="embeddings")

    ckpt_emb = tf.train.Checkpoint(embeddings=emb_storage)
    ckpt_emb.save(os.path.join(cfg.dirs.emb_log, "embedding.ckpt"))
    metadata = os.path.join(cfg.dirs.emb_log, 'metadata.tsv')
    with open(metadata, 'w') as metadata_file:
        for row in pids:
            metadata_file.write('%d\n' % int(row))


    # Generar una visualización del dataset
    # Cogemos las imagenes del datas set.
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name =  'embeddings/.ATTRIBUTES/VARIABLE_VALUE'
    embedding.metadata_path = metadata
    embedding.sprite.image_path = sprite_file
    embedding.sprite.single_image_dim.extend(thumb_size)
    # Definimos la ubicación donde se almacena los logs de Tensorboard
    # emb_summary_writer = tf.summary.create_file_writer(cfg.dirs.emb_log)
    projector.visualize_embeddings(cfg.dirs.emb_log, config)



if __name__ == '__main__':

    args = parser.parse_args()

    config_file = args.config
    batch_size = args.batch_size
    with open(config_file) as f:
        config = json.load(f, object_hook=lambda d: Namespace(**d))

    config = directories(config, state='emb')
    config.batch_size = batch_size

    print (config.batch_size)
    main(config)

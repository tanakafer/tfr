import os
import configuration as const
from datetime import timedelta, datetime
from argparse import Namespace
from utils import os_utils



def directories(cfg, state='train'):
    # Seleccionamos los directorios dependiendo del dataset

    if cfg.dataset == 'cub':
        dataset_dir = 'CUB_200_2011'
        dataset_file = 'cub_train'
        test_file ='cub_test'
    elif cfg.dataset == 'bags':
        dataset_dir = 'BAGS'
        dataset_file = 'bags_train'
        test_file ='bags_test'
    elif cfg.dataset == 'bags_40':
        dataset_dir = 'BAGS_40'
        dataset_file = 'bags_train'
        test_file ='bags_test'
    elif cfg.dataset == 'bags_40_v2':
        dataset_dir = 'BAGS_40_v2'
        dataset_file = 'bags_train'
        test_file ='bags_test'
    else:
        raise NotImplementedError('El dataset {} no existe'.format(cfg.dataset))

    cfg.dirs=Namespace()
    cfg.dirs.csv_file= os.path.join(const.dataset_dir, dataset_dir, dataset_file + '.csv')
    cfg.dirs.images= os.path.join(const.dataset_dir,dataset_dir, 'images')
    cfg.dirs.trained_models = const.trained_models_dir

    # Creamos el nombre para la almacenar la información del emb_modelo
    if cfg.model.fit.loss == "angular_loss":
        exp_name = [cfg.dataset, cfg.model.name, cfg.model.head,cfg.model.fit.optimizer,cfg.model.fit.loss, 'alpha_{}'.format(cfg.model.fit.alpha)]
    else:
        exp_name = [cfg.dataset, cfg.model.name, cfg.model.head,cfg.model.fit.optimizer,cfg.model.fit.loss, 'm_{}'.format(cfg.model.fit.margin)]
    cfg.model_name = '_'.join(exp_name)

    # El directorio de los checkpoint
    cfg.dirs.checkpoint = os.path.join(const.experiment_root_dir, cfg.model_name ,'tf_ckpts')
    os_utils.touch_dir(cfg.dirs.checkpoint)


    # El directorio de los modelos entrenados
    cfg.dirs.trained = os.path.join(const.trained_models_dir,cfg.model_name)
    os_utils.touch_dir(cfg.dirs.trained)

    # Definimos el directorio donde almacenar los datos de Tensorboard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.dirs.train_log = os.path.join(const.tensorboard_dir, cfg.model_name,'train', current_time)
    os_utils.touch_dir(cfg.dirs.train_log)
    cfg.dirs.test_log = os.path.join(const.tensorboard_dir, cfg.model_name,'test', current_time)
    os_utils.touch_dir(cfg.dirs.train_log)
    cfg.dirs.eval_log = os.path.join(const.tensorboard_dir, cfg.model_name,'eval', current_time)
    os_utils.touch_dir(cfg.dirs.eval_log)

    # Definimos la ubicación de los log
    cfg.dirs.logs = os.path.join(const.experiment_root_dir, "train")

    # Cuando vayamos a testear el modelo
    if state == 'test':
        cfg.dirs.embeddings= const.embeddings_dir
        cfg.dirs.embeddings_file = os.path.join(cfg.dirs.embeddings, cfg.model_name +".h5")
        cfg.dirs.test_file= os.path.join(const.dataset_dir, dataset_dir, test_file + '.csv')
        #Creamos el directorios
        os_utils.touch_dir(cfg.dirs.embeddings)
    if state == 'emb':
        cfg.dirs.embeddings= const.embeddings_dir
        cfg.dirs.embeddings_file = os.path.join(cfg.dirs.embeddings, cfg.model_name +".h5")
        cfg.dirs.emb_log = os.path.join(const.tensorboard_dir, cfg.model_name,'emb', current_time)
        os_utils.touch_dir(cfg.dirs.emb_log)
        #Creamos el directorios
        os_utils.touch_dir(cfg.dirs.embeddings)
    elif state =='production':
        if cfg.dataset == 'cub':
            labels_file = 'classes.txt'
        elif cfg.dataset == 'bags':
            labels_file = 'bags_labels.csv'
        # Directorio para recuperar los nombres de las etiquetas de las imágenes
        cfg.dirs.labels_file= os.path.join(const.dataset_dir, dataset_dir, labels_file)
        cfg.dirs.embeddings= const.embeddings_dir
        cfg.dirs.embeddings_file = os.path.join(cfg.dirs.embeddings, cfg.model_name +".h5")
        cfg.dirs.cbir_log = os.path.join(const.tensorboard_dir, cfg.model_name,'cbir', current_time)
        os_utils.touch_dir(cfg.dirs.cbir_log)




    return cfg

import os
# Directorio Raiz
data_dir ='/mnt/data'

# Donde están los datasets
dataset_dir = os.path.join(data_dir, 'datasets')

# Donde se almacena los modelos preentrenados
trained_models_dir = os.path.join(data_dir, 'pretrained')

# Donde se alamancena los checkpoint
experiment_root_dir = os.path.join(data_dir, 'checkpoints/retrieval_models')


# Donde se almacena las base de datos
# de los vectores de características
embeddings_dir = os.path.join(data_dir,'emb')

# Donde se almacena los logs de tensorboard
tensorboard_dir = os.path.join(data_dir,'logs/gradient_tape')

# This file contains select utilities from Ahmed Taha's project, the complete
# project can be found at https://github.com/ahmdtaha/tf_retrieval_baseline.git
#
# The content of this file is copyright Ahmed Taha. You may only re-use
# parts of it by keeping the following comment above it:
#
# This is taken from Lucas Beyer's toolbox© found at
#    https://github.com/ahmdtaha/tf_retrieval_baseline.git
# and may only be redistributed and reused by keeping this notice

import tensorflow as tf

class FC1024Head(tf.keras.Model):
    def __init__(self, cfg):
        super(FC1024Head, self).__init__()
        self.h_1024 = tf.keras.layers.Dense(1025, activation=None,
                                          kernel_initializer=tf.keras.initializers.Orthogonal())
        self.batch_norm = tf.keras.layers.BatchNormalization(
            momentum = 0.9,
            epsilon=1e-5,
            scale=True,
        )
        self.head = tf.keras.layers.Dense(cfg.model.embedding_dim, activation=None,
                                                    kernel_initializer=tf.keras.initializers.Orthogonal())
    def call(self, inputs):
        h1 = tf.keras.backend.relu(self.batch_norm(self.h_1024(inputs)))
        return self.head(h1)


class DirectHead(tf.keras.Model):
    def __init__(self, cfg):
        super(DirectHead, self).__init__()
        self.head = tf.keras.layers.Dense(cfg.model.embedding_dim, activation=None,
                                                    kernel_initializer=tf.keras.initializers.Orthogonal())
    def call(self, inputs):
        return self.head(inputs)

class EmbeddingModel(tf.keras.Model):

    def __init__(self, cfg):
        super(EmbeddingModel, self).__init__()
        self.cfg = cfg
        if cfg.model.name == 'inception_v1':
            self.base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False)
            self.preprocess_input = tf.keras.applications.xception.preprocess_input
        elif cfg.model.name == 'inception_v3':
            self.base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
            self.preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        elif cfg.model.name == 'resnet_v1_50':
            self.base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
            self.preprocess_input = tf.keras.applications.resnet.preprocess_input
        elif cfg.model.name == 'densenet169':
            self.base_model = tf.keras.applications.DenseNet169(weights='imagenet', include_top=False)
            self.preprocess_input = tf.keras.applications.densenet.preprocess_input
        elif cfg.model.name == 'vgg16':
            self.base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
            self.preprocess_input = tf.keras.applications.vgg16.preprocess_input
        else:
            raise NotImplementedError('Invalid model.name {}'.format(cfg.model.name))



        self.spatial_pooling = tf.keras.layers.GlobalAvgPool2D()
        if 'direct' in cfg.model.head:
            self.embedding_head = DirectHead(cfg)
        elif 'fc1024' in cfg.model.head:
            self.embedding_head = FC1024Head(cfg)
        else:
            raise NotImplementedError('Invalid model.head {}'.format(cfg.model.head))


        self.l2_embedding = 'normalize' in cfg.model.head
        print(self.l2_embedding)



    def call(self, images):

        # Como base el modelo base
        base_model_output = self.base_model(images)
        # Obtenemos las características de las últimas capas
        base_model_output_pooled = self.spatial_pooling(base_model_output)
        # Obtenemosel vector de características con el tamaño  definido
        batch_embedding = self.embedding_head(base_model_output_pooled )
        # Realizamos la norma de orden 2 si procede
        if self.l2_embedding:
            return_batch_embedding = tf.nn.l2_normalize(batch_embedding, -1)
        else:
            return_batch_embedding = batch_embedding
        return return_batch_embedding

#  Tensorflow retrieval

Este repositorio proporciona una base para la creación de un sistema CBIR.
Este código está basado en [ahmdtaha/tf_retrieval_baseline](https://github.com/ahmdtaha/tf_retrieval_baseline.git)


# Requerimientos

- Python 3+ [Comprobado en 3.6.9/3.7.5]
- Tensorfow 2.0 [Comprobado con 2.1.0rc0/2.1.0+nv]


# Utilización

1. Renombra el fichero 'configuration_sample.py' a 'configuration.py'
2. Edita el nuevo fichero 'configuration.py'
3. Establece el campo 'data_dir' al directorio donde se almacenará los dataset y los resultados del entrenamiento

Hay cuatro ficheros a ejecutar.

- **train.py**: Para realizar el entrenamiento es necesario definir la red y los parámetros. Esto se define por un fichero de configuración que se encuentra en el directorio "trainings".
En el presente trabajo se ha entrenado una red ResNet Versión 1 de 50 capas, con las funciones de coste Angular Loss a 30 y 45 gradios y Semi-hard Triplet con margen 0.2 y 0.5.

En la siguiente tabla se define los ficheros de configuración según el dataset utilizado y la función de coste.

| Dataset   | Función Coste                | Fichero                                                                 |
|-----------|------------------------------|-------------------------------------------------------------------------|
| Cat-40    | Angular Loss  alpha=30       | bags_40_resnet_v1_50_normalize_adam_angular_loss_alpha_30.json          |
| Cat-40    | Angular Loss  alpha=45       | bags_40_resnet_v1_50_normalize_adam_angular_loss_alpha_45.json          |
| Cat-40    | Semi-hard Triplet margin=0.2 | bags_40_resnet_v1_50_normalize_adam_triplet_semihard_loss_m_0.2.json    |
| Cat-40    | Semi-hard Triplet margin=0.2 | bags_40_resnet_v1_50_normalize_adam_triplet_semihard_loss_m_0.5.json    |
| Cat-40-V2 | Angular Loss  alpha=30       | bags_40_v2_resnet_v1_50_normalize_adam_angular_loss_alpha_30.json       |
| Cat-40-V2 | Angular Loss  alpha=45       | bags_40_v2_resnet_v1_50_normalize_adam_angular_loss_alpha_45.json       |
| Cat-40-V2 | Semi-hard Triplet margin=0.2 | bags_40_v2_resnet_v1_50_normalize_adam_triplet_semihard_loss_m_0.2.json |
| Cat-40-V2 | Semi-hard Triplet margin=0.2 | bags_40_v2_resnet_v1_50_normalize_adam_triplet_semihard_loss_m_0.5.json |
| Cat-13    | Angular Loss  alpha=30       | bags_resnet_v1_50_normalize_adam_angular_loss_alpha_30.json             |
| Cat-13    | Angular Loss  alpha=45       | bags_resnet_v1_50_normalize_adam_angular_loss_alpha_45.json             |
| Cat-13    | Semi-hard Triplet margin=0.2 | bags_resnet_v1_50_normalize_adam_triplet_semihard_loss_m_0.2.json       |
| Cat-13    | Semi-hard Triplet margin=0.2 | bags_resnet_v1_50_normalize_adam_triplet_semihard_loss_m_0.5.json       |

Como ejemplo, para entrenar el modelo con el dataset de 40 categorías versión 1 con  la función de coste angular a 30 grados tendríamos que ejecutar el siguiente

```
   python train.py -c trainings/bags_40_v2_resnet_v1_50_normalize_adam_angular_loss_alpha_30.json
```
- **embed.py:** Este fichero lo utilizamos para crear la base de datos de nuestro sistema CBIR según el modelo entrenado en el paso anterior

Con el ejemplo anterior tendríamos que ejecutar:

```
   python embed.py -c trainings/bags_40_v2_resnet_v1_50_normalize_adam_angular_loss_alpha_30.json
```

- test.py: Para realizar una prueba. Coge una imagen aleatoria del conjunto de test del dataset y devuelve las imágenes similares.

Siguiendo con el ejemplo si quisieramos recuperar 10 imágenes desde la imágen aleatorias del conjunto  de test tendríamos que ejecutar

```
  python test.py -c trainings/bags_40_v2_resnet_v1_50_normalize_adam_angular_loss_alpha_30.json -r 10
```


- **evaluation.py**: Para realizar una evaluación del modelo de recuperación calculando la métrica mAP (Mean Average Precision).

Siguiendo con el ejemplo para evaluar el modelo recuperando 10 imágenes con 100 intentos tendríamos que ejecutar:

```
  python evaluation.py -c trainings/bags_40_v2_resnet_v1_50_normalize_adam_angular_loss_alpha_30.json -t 100 -r 10
```


- **cbir.py**: Para realizar una prueba con cualquier imagen de un fichero y que devuelva las 10 imágenes más similares.

```
  python cbir.py -c trainings/bags_40_v2  _resnet_v1_50_normalize_adam_angular_loss_alpha_30.json -r 10 /mnt/data/IMG_saddle_calskin.jpg
```

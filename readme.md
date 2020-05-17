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


Hay tres ficheros a ejecutar.

- train.py: Para realizar el entrenamiento
- test.py: Para realizar una prueba. Coge una imagen aleatoria del conjunto de test del dataset y devuelve las imágenes similares.
- cbir.py: Para realizar una prueba con cualquier desde un fichero y devuelve las imágenes similares.
- evaluation.py: Para realizar una evaluación del modelo de recuperación calculando la métrica mAP (Mean Average Precision)

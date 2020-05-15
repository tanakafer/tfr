import numpy as np
import os
import tensorflow as tf
import io
import matplotlib.pyplot as plt
# Basado en el proyecto  https://github.com/ahmdtaha/tf_retrieval_baseline.git

# Basado en el proyecto  https://github.com/ahmdtaha/tf_retrieval_baseline.git

def load_dataset(csv_file, image_root, fail_on_missing=True):
    """ Loads a dataset .csv file, returning PIDs and FIDs.

    PIDs are the "person IDs", i.e. class names/labels.
    FIDs are the "file IDs", which are individual relative filenames.

    Args:
        csv_file (string, file-like object): The csv data file to load.
        image_root (string): The path to which the image files as stored in the
            csv file are relative to. Used for verification purposes.
            If this is `None`, no verification at all is made.
        fail_on_missing (bool or None): If one or more files from the dataset
            are not present in the `image_root`, either raise an IOError (if
            True) or remove it from the returned dataset (if False).

    Returns:
        (pids, fids) a tuple of numpy string arrays corresponding to the PIDs,
        i.e. the identities/classes/labels and the FIDs, i.e. the filenames.

    Raises:
        IOError if any one file is missing and `fail_on_missing` is True.
    """
    dataset = np.genfromtxt(csv_file, delimiter=',',dtype='|U')
    #dataset = np.genfromtxt(csv_file, delimiter=',', dtype='|U')
    pids, fids = dataset.T
    # Possibly check if all files exist
    if image_root is not None:
        missing = np.full(len(fids), False, dtype=bool)
        for i, fid in enumerate(fids):
            missing[i] = not os.path.isfile(os.path.join(image_root, fid))

        missing_count = np.sum(missing)
        if missing_count > 0:
            if fail_on_missing:
                raise IOError('Using the `{}` file and `{}` as an image root {}/'
                            '{} images are missing'.format(
                                csv_file, image_root, missing_count, len(fids)))
            else:
                print('[Warning] removing {} missing file(s) from the'
                    ' dataset.'.format(missing_count))
                # We simply remove the missing files.
                fids = fids[np.logical_not(missing)]
                pids = pids[np.logical_not(missing)]

    return pids, fids

# Basado en el proyecto  https://github.com/ahmdtaha/tf_retrieval_baseline.git

def sample_k_fids_for_pid(pid, all_fids, all_pids, batch_k):
    """ Given a PID, select K FIDs of that specific PID. """
    possible_fids = tf.boolean_mask(all_fids, tf.math.equal(all_pids, pid))

    # The following simply uses a subset of K of the possible FIDs
    # if more than, or exactly K are available. Otherwise, we first
    # create a padded list of indices which contain a multiple of the
    # original FID count such that all of them will be sampled equally likely.
    count = tf.shape(possible_fids)[0]
    padded_count = tf.cast(tf.math.ceil(batch_k / tf.dtypes.cast(count, tf.dtypes.float32)), tf.dtypes.int32) * count
    full_range = tf.math.mod(tf.range(padded_count), count)

    # Sampling is always performed by shuffling and taking the first k.
    shuffled = tf.random.shuffle(full_range)
    selected_fids = tf.gather(possible_fids, shuffled[:batch_k])

    return selected_fids, tf.fill([batch_k], pid)

# Basado en el proyecto  https://github.com/ahmdtaha/tf_retrieval_baseline.git

def fid_to_image(fid, pid, image_root, image_size):
    # fid = tf.Print(fid,[fid, pid],'fid ::')
    """ Loads and resizes an image given by FID. Pass-through the PID. """
    # Since there is no symbolic path.join, we just add a '/' to be sure.
    image_encoded = tf.io.read_file(tf.strings.reduce_join([image_root, '/', fid]))

    # tf.image.decode_image doesn't set the shape, not even the dimensionality,
    # because it potentially loads animated .gif files. Instead, we use either
    # decode_jpeg or decode_png, each of which can decode both.
    # Sounds ridiculous, but is true:
    # https://github.com/tensorflow/tensorflow/issues/9356#issuecomment-309144064
    image_decoded = tf.io.decode_jpeg(image_encoded, channels=3)
    image_resized = tf.image.resize(image_decoded, image_size)

    return image_resized, fid, pid

# Buscamos el pid de un fid

def find_pid(fid, all_fids, all_pids):
    pid = tf.boolean_mask(all_pids, tf.math.equal(all_fids, fid))[0]
    return fid, pid
# Creamos un boolean label.
# Definimos el label de  la respuesta como 1 si coincide con la query,
#                                          0  si no conincide con la query

def boolean_label(pid, idx, distance, query_label):
    if tf.math.equal(pid, query_label):
        boolean_label = 1
    else:
        boolean_label = 0

    return pid,idx, distance, boolean_label



def remove_embs(embs, pid, idx, distance):
    return  pid,idx, distance

# Obtenemos las imaǵenes más próximas
def calculate_distances( embs, pid, idx, query, type='Euclidian'):


    if type == 'Euclidian':
        distance=tf.math.reduce_euclidean_norm(
                            tf.math.add(embs, -query)
                        )
    return embs, pid, idx, distance


# Dibuja una imágen en el sumary image_encoded
def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


 # Representamos las imágen origianl
 # enfretada a las imágenes Recuperadas

def show_images(image_orginal, fid_original, pid_original,
                images, fids, pids, sorted_distances,
                columns=4,
                ap = 0,
                aps = np.zeros(1) ):

    n_retrievals = len(fids)
    rows = round(n_retrievals / columns )
    if n_retrievals % columns !=0:
        rows = rows +2
    else:
        rows = rows +1


    # Dibujamos la imágen original
    pid= pid_original[0]
    image = image_orginal.numpy().astype('float32')[0] /255

    figure = plt.figure(figsize=(10,10))
    title = "Original Label: {} ".format(pid)
    plt.subplot(rows, columns, 1, title=title)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)

    # Dibujamos un resumen de la respuesta
    
    title = "Resumen ap: {:.2f}".format(ap)
    plt.subplot(rows, columns, 2, title=title)
    x = np.arange(aps.shape[0])+1
    plt.plot(x, aps,  'o-')
    plt.show()

    for i in range(n_retrievals):
        # Start next subplot.
        pid= pids[i]
        fid = fids[i]
        distance = sorted_distances[i]
        #Normalizamos la imágen
        image = images.numpy().astype('float32')[i] /255
        #title = "Label: {} Distance: {} Fichero: {}".format(pid, distance, fid)
        #title = "Label: {} Distance: {}".format(pid, distance)
        title = " R{} Label: {} ".format( i+1, pid)
        plt.subplot(rows, columns, i + (columns+1), title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

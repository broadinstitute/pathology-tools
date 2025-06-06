'''
From https://github.com/tsc2017/Frechet-Inception-Distance
Code derived from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py

Usage:
    Call get_fid(images1, images2)
Args:
    images1, images2: Numpy arrays with values ranging from 0 to 255 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary.
    dtype of the images is recommended to be np.uint8 to save CPU memory.
Returns:
    Frechet Inception Distance between the two image distributions.
'''

import tensorflow as tf
import os
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
import pickle

if float('.'.join(tf.__version__.split('.')[:2])) < 1.15:
    tfgan = tf.contrib.gan
else:
    import tensorflow_gan as tfgan
session = tf.compat.v1.InteractiveSession()
# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 64

# Run images through Inception.
inception_images = tf.compat.v1.placeholder(tf.float32, [None, 3, None, None])
activations1 = tf.compat.v1.placeholder(tf.float32, [None, None], name='activations1')
activations2 = tf.compat.v1.placeholder(tf.float32, [None, None], name='activations2')
fcd = tfgan.eval.frechet_classifier_distance_from_activations(activations1, activations2)


def inception_activations(images=inception_images, num_splits=1):
    # debug
    print(f'EXECUTING inception_activations()\nimages.shape={images.shape}')
    images = tf.transpose(images, [0, 2, 3, 1])
    print(f'(after tf.transpose()) type(images)={type(images)}; images.shape={images.shape}')
    size = 299
    images = tf.compat.v1.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
    activations = tf.map_fn(
        fn=functools.partial(tfgan.eval.run_inception, output_tensor='pool_3:0'),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=8,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    activations = array_ops.concat(array_ops.unstack(activations), 0)
    return activations


activations = inception_activations()


def get_inception_activations(inps):
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    act = np.zeros([inps.shape[0], 2048], dtype=np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] / 255. * 2 - 1
        act[i * BATCH_SIZE: i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = session.run(activations, feed_dict={
            inception_images: inp})
    return act


def activations2distance(act1, act2):
    return session.run(fcd, feed_dict={activations1: act1, activations2: act2})

def dataset_prep_from_numpy(images, save_path=None):
    """
    FID calculation requires images with pixels in [0, 255], and of shape (n_samples, 3, H, W)
    data preparation step meant to mimic the preparation that we use in evaluation_tools/evaluate.py
    but appears that inception_activations() may be written to perform this step
    --> while testing going to mimic the working evaluation pipeline as closely as possible,
    --> but may be better practice to shift to use of inception_activations() if possible
    """
    # scaling the images to the right range of pixel values for FID calculation (train set and PathologyGAN output has pixels \in (0,1))
    processed_images = np.multiply(images, 255).astype(np.uint8)
    # debug
    print(f'RESHAPING DATA FROM {processed_images.shape} to {(len(processed_images), 3, 224, 224)}')
    processed_images = np.reshape(processed_images, (len(processed_images), 3, 224, 224))
    # **NB: inception_activations must be executed because InceptionNet requires dimensions of 299,299,
    #       but dataset_prep() output images of dimension 224,224 before calling get_fid()
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(processed_images, f, protocol=pickle.HIGHEST_PROTOCOL)

    return processed_images

def get_fid(images1, images2):
    assert (type(images1) == np.ndarray)
    assert (len(images1.shape) == 4)
    assert (images1.shape[1] == 3)
    assert (np.min(images1[0]) >= 0 and np.max(images1[0]) > 10), 'Image values should be in the range [0, 255]'
    assert (type(images2) == np.ndarray)
    assert (len(images2.shape) == 4)
    assert (images2.shape[1] == 3)
    assert (np.min(images2[0]) >= 0 and np.max(images2[0]) > 10), 'Image values should be in the range [0, 255]'
    assert (images1.shape == images2.shape), 'The two numpy arrays must have the same shape'
    print('Calculating FID with %i images from each distribution' % (images1.shape[0]))
    start_time = time.time()
    act1 = get_inception_activations(images1)
    act2 = get_inception_activations(images2)
    fid = activations2distance(act1, act2)
    print('FID calculation time: %f s' % (time.time() - start_time))
    return fid
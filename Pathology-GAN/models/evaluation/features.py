import os
import tensorflow as tf
import numpy as np
import h5py
import random
import shutil
import tensorflow.contrib.gan as tfgan
from models.generative.utils import *
from data_manipulation.utils import *
# for saving latent representations of generated images
import pickle


# Gather real samples from train and test sets for FID and other scores.
def real_samples(data, data_output_path, num_samples=5000):
    path = os.path.join(data_output_path, 'evaluation')
    path = os.path.join(path, 'real')
    path = os.path.join(path, data.dataset)
    path = os.path.join(path, data.marker)
    res = 'h%s_w%s_n%s' % (data.patch_h, data.patch_w, data.n_channels)
    path = os.path.join(path, res)
    if not os.path.isdir(path):
        os.makedirs(path)

    batch_size = data.batch_size
    images_shape =  [num_samples] + [data.patch_h, data.patch_w, data.n_channels]

    hdf5_sets_path = list()
    dataset_sets_path = [data.hdf5_train, data.hdf5_validation, data.hdf5_test]
    dataset_sets = [data.training, data.validation, data.test]
    for i_set, set_path in enumerate(dataset_sets_path):
        set_data = dataset_sets[i_set]
        if set_data is None:
            continue
        type_set = set_path.split('_')[-1]
        type_set = type_set.split('.')[0]   
        img_path = os.path.join(path, 'img_%s' % type_set)
        if not os.path.isdir(img_path):
            os.makedirs(img_path)

        hdf5_path_current = os.path.join(path, 'hdf5_%s_%s_images_%s_real.h5' % (data.dataset, data.marker, type_set))
        hdf5_sets_path.append(hdf5_path_current)

        if os.path.isfile(hdf5_path_current):
            print('H5 File Image %s already created.' % type_set)
            print('\tFile:', hdf5_path_current)
        else:
            print('H5 File Image %s.' % type_set)
            print('\tFile:', hdf5_path_current)

            hdf5_img_real_file = h5py.File(hdf5_path_current, mode='w')
            img_storage = hdf5_img_real_file.create_dataset(name='images', shape=images_shape, dtype=np.float32)

            possible_samples = len(set_data.images)
            random_samples = list(range(possible_samples))
            random.shuffle(random_samples)

            ind = 0
            for index in random_samples[:num_samples]:
                img_storage[ind] = set_data.images[index]
                plt.imsave('%s/real_%s_%s.png' % (img_path, type_set, ind), set_data.images[index])
                ind += 1
            print('\tNumber of samples:', ind)

    return hdf5_sets_path


# Extract Inception-V1 features from images in HDF5.
def inception_tf_feature_activations(hdf5s, input_shape, batch_size):
    images_input = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape, name='images')
    images = 2*images_input
    images -= 1
    images = tf.image.resize_bilinear(images, [299, 299])
    out_incept_v3 = tfgan.eval.run_inception(images=images, output_tensor='pool_3:0')

    hdf5s_features = list()
    with tf.Session() as sess:
        for hdf5_path in hdf5s:
            # Name handling.
            # hdf5_feature_path = hdf5_path.replace('_images_','_features_')
            hdf5_feature_path = hdf5_path.split('.h5')[0] + '_features.h5'
            hdf5s_features.append(hdf5_feature_path)
            if os.path.isfile(hdf5_feature_path):
                print('H5 File Feature already created.')
                print('\tFile:', hdf5_feature_path)
                continue
            hdf5_img_file = h5py.File(hdf5_path, mode='r')
            flag_images = False
            hdf5_features_file = h5py.File(hdf5_feature_path, mode='w')
            for key in list(hdf5_img_file.keys()):
                if 'images' in key:
                    flag_images = True
                    storage_name = key.replace('images', 'features')
                    images_storage = hdf5_img_file[key]
                    
                    num_samples = images_storage.shape[0]
                    batches = int(num_samples/batch_size)
                    features_shape = (num_samples, 2048)
                    features_storage = hdf5_features_file.create_dataset(name=storage_name, shape=features_shape, dtype=np.float32)

                    print('Starting features extraction...')
                    print('\tImage File:', hdf5_path)
                    print('\t\tImage type:', key)
                    ind = 0
                    for batch_num in range(batches):
                        batch_images = images_storage[batch_num*batch_size:(batch_num+1)*batch_size]
                        if np.amax(batch_images) > 1.0:
                            batch_images = batch_images/255.
                        activations = sess.run(out_incept_v3, {images_input: batch_images})
                        features_storage[batch_num*batch_size:(batch_num+1)*batch_size] = activations
                        ind += batch_size
                    print('\tFeature File:', hdf5_feature_path)
                    print('\tNumber of samples:', ind)
            if not flag_images:
                os.remove(hdf5_features_file)
    return hdf5s_features

# Generate random samples from a model, it also dumps a sprite image width them.
def generate_samples_epoch(session, model, data_shape, epoch, evaluation_path, num_samples=5000, batches=50):
    epoch_path = os.path.join(evaluation_path, 'epoch_%s' % epoch)
    check_epoch_path = os.path.join(epoch_path, 'checkpoints')
    checkpoint_path = os.path.join(evaluation_path, '../checkpoints')
    
    os.makedirs(epoch_path)
    shutil.copytree(checkpoint_path, check_epoch_path)

    if model.conditional:
        runs = ['postive', 'negative']
    else:
        runs = ['unconditional']

    for run in  runs:

        hdf5_path = os.path.join(epoch_path, 'hdf5_epoch_%s_gen_images_%s.h5' % (epoch, run))
        
        # H5 File.
        img_shape = [num_samples] + data_shape
        hdf5_file = h5py.File(hdf5_path, mode='w')
        storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)

        ind = 0
        while ind < num_samples:
            if model.conditional:
                label_input = model.label_input
                if 'postive' in run:
                    labels = np.ones((batches, 1))
                else:
                    labels = np.zeros((batches, 1))
                labels = tf.keras.utils.to_categorical(y=labels, num_classes=2)
            else:
                label_input=None
                labels=None
            gen_samples, _ = show_generated(session=session, z_input=model.z_input, z_dim=model.z_dim, output_fake=model.output_gen, label_input=label_input, labels=labels, n_images=batches, show=False)

            for i in range(batches):
                if ind == num_samples:
                    break
                storage[ind] = gen_samples[i, :, :, :]
                ind += 1

# Attempt at generating latents by interpolating between two exemplars
# Here are the centroids of three-example clusters of low tumor density and high tumor density images with z_dim=200
#high_density_centroid = np.array([ 0.39      ,  0.87666667,  0.38333333,  1.06666667,  0.95666667,
#                                -0.2       , -0.00666667, -0.73666667,  0.3       , -0.48666667,
#                                 0.09333333, -0.21      , -1.27      , -0.83      ,  0.74      ,
#                                 0.11      ,  0.26      ,  0.79      ,  0.10333333,  1.05666667,
#                                -0.12      ,  0.3       , -0.07      , -0.66      ,  1.2       ,
#                                 1.31666667,  0.46      , -0.01666667,  0.61666667,  0.69666667,
#                                -0.50333333, -0.10666667, -0.11      ,  1.35      ,  1.62333333,
#                                 0.36      , -0.41333333,  0.35666667, -0.38333333,  0.74      ,
#                                -0.33333333, -0.11      ,  0.00333333, -0.16666667,  0.03333333,
#                                -0.21333333,  0.05      , -0.52333333,  0.60333333, -0.57333333,
#                                -0.09      ,  0.18333333,  0.86      , -0.07666667, -1.15333333,
#                                 0.5       , -0.27      , -0.75333333,  0.00666667,  0.07      ,
#                                 0.78      ,  0.77666667, -0.54666667, -0.22666667, -0.26666667,
#                                 0.25      ,  1.03333333,  0.40333333,  0.25333333, -0.58      ,
#                                 0.03666667, -0.31666667,  0.96333333,  0.35      ,  1.04666667,
#                                 0.25666667, -0.65      , -1.11      ,  0.58333333, -0.17      ,
#                                -0.90666667, -0.06666667,  0.49333333,  0.74666667, -0.05      ,
#                                -0.38333333,  0.35      , -0.23333333,  0.67      , -0.36      ,
#                                -0.77666667,  1.37      ,  0.07      , -0.05      , -1.13666667,
#                                -0.11333333, -0.88      , -0.60666667,  0.07666667, -1.86      ,
#                                 0.61666667, -0.79666667, -0.55666667, -0.02666667, -0.47666667,
#                                 0.10666667, -1.04333333, -0.51666667, -1.35333333, -0.45666667,
#                                 0.08333333, -0.23333333,  0.46333333, -0.58      , -0.27666667,
#                                 0.10666667,  0.37666667, -0.15666667, -0.57333333,  0.31333333,
#                                -0.59333333, -0.26333333,  1.17      , -0.62      , -0.44      ,
#                                -0.20666667,  0.44666667, -0.66666667, -0.51333333,  0.69333333,
#                                -0.20666667, -0.01333333, -0.16333333,  0.17      , -0.13      ,
#                                 0.45333333,  0.52666667, -0.77333333, -0.33333333, -0.52333333,
#                                -1.03      , -0.37      , -1.24333333,  0.67666667,  0.27      ,
#                                 0.46666667, -0.51333333, -0.33333333, -0.15      , -0.91333333,
#                                -0.08      , -0.26      , -0.79333333, -0.16333333, -0.24666667,
#                                 0.51      ,  0.34333333, -0.35333333, -0.32      , -0.79333333,
#                                 0.91333333,  1.19666667, -0.48333333,  0.01333333, -0.39333333,
#                                -0.70333333, -1.39      , -0.30333333, -0.08      , -0.62333333,
#                                 0.98      ,  0.48666667,  0.04333333,  0.05333333, -1.23333333,
#                                -0.21      ,  0.83333333,  0.42      , -0.53333333, -0.21333333,
#                                 0.64666667, -0.96666667,  0.46666667,  0.94333333, -0.63333333,
#                                -0.36333333,  0.32333333,  0.64666667, -0.30666667,  0.01      ,
#                                -0.30333333, -0.06      ,  0.09333333,  0.31      , -0.05      ,
#                                 0.41333333, -0.18      ,  0.19666667, -0.13      ,  0.98333333])
#
#low_density_centroid = np.array([-0.28666667,  0.33      ,  1.06666667, -0.26333333, -0.74333333,
#                             0.32666667,  0.47      ,  0.13      , -0.45      , -0.14      ,
#                             0.18333333, -0.19666667, -0.40666667,  0.39333333,  0.12333333,
#                            -0.33333333,  0.45666667, -0.27333333, -0.25      ,  0.62      ,
#                            -0.40666667, -0.78      ,  0.78333333, -0.21333333, -0.27      ,
#                             0.00333333,  0.04      , -0.28333333, -1.33666667,  1.29      ,
#                            -0.93      , -0.84      ,  0.06333333,  0.62666667, -1.26      ,
#                            -1.42333333,  1.22333333, -0.57333333,  0.45666667,  0.87666667,
#                            -0.59      ,  0.49666667,  0.81333333,  0.34      , -0.51333333,
#                            -0.02333333, -0.40666667, -1.13666667,  0.49666667, -0.60333333,
#                            -0.22666667, -0.27333333, -0.04666667,  0.11      ,  0.46      ,
#                            -1.22      , -0.00333333,  0.99666667, -0.87333333, -0.02333333,
#                            -0.55333333, -1.19333333, -0.38333333, -0.25333333,  0.94333333,
#                             0.15333333,  1.06      ,  0.18333333,  0.24      ,  1.00666667,
#                            -0.46333333,  0.08333333,  0.07666667,  0.31      ,  1.35333333,
#                             0.56666667,  0.01666667, -0.71666667,  1.17666667,  0.43      ,
#                             1.31666667,  0.27333333,  0.64333333, -0.35666667, -0.01333333,
#                             0.2       ,  0.31333333,  0.47666667,  0.12666667,  1.30333333,
#                             0.27666667, -0.39666667, -0.18666667, -0.39666667, -0.2       ,
#                            -0.18333333, -0.14333333, -1.21333333,  0.12666667, -0.83666667,
#                             0.04333333,  0.75      ,  1.28      ,  0.18666667,  0.71333333,
#                            -1.35      , -0.33666667,  0.29      , -0.63666667,  0.66666667,
#                             0.09666667, -0.45666667, -0.67666667, -0.02      , -0.01666667,
#                            -0.78      ,  0.72666667, -0.6       , -0.9       ,  0.27333333,
#                             0.20333333,  0.30333333,  0.89333333, -0.44333333,  0.3       ,
#                            -0.17333333, -0.05333333,  0.15666667,  0.49666667,  0.26      ,
#                            -1.24333333,  0.2       , -0.02333333,  0.37666667,  0.28      ,
#                             0.78      , -0.31333333,  0.06      ,  0.72      , -0.20333333,
#                             0.31666667, -0.50666667,  0.87333333,  0.60333333,  0.00333333,
#                            -0.00333333, -0.18666667, -0.56      ,  0.17666667, -0.72666667,
#                             0.11666667,  0.10333333,  0.78666667, -0.72      ,  0.37666667,
#                            -0.21      , -0.82666667, -0.17      , -0.57666667,  1.17      ,
#                             0.30333333, -0.60666667, -0.30333333,  0.71666667,  0.44333333,
#                            -1.16666667, -0.11333333, -0.29      ,  0.85      , -0.13      ,
#                             0.85333333, -1.04      ,  0.43333333, -0.54      ,  0.46666667,
#                            -0.14      , -1.34666667, -0.12      , -0.15      ,  0.65666667,
#                             0.22666667, -0.91666667,  0.34      , -0.50333333,  0.19      ,
#                            -0.42333333,  0.52666667, -0.48333333, -0.00666667, -0.19666667,
#                            -0.34      , -0.22666667,  0.19      ,  0.58      ,  0.02      ,
#                            -0.10666667,  0.63333333, -0.65333333, -0.71666667, -0.09333333])

# Generate sampeles from PathologyGAN, no encoder.
def generate_samples_from_checkpoint(model, data, data_out_path, checkpoint, num_samples=5000, batches=50, exemplar1=None, exemplar2=None):
    path = os.path.join(data_out_path, 'evaluation')
    path = os.path.join(path, model.model_name)
    path = os.path.join(path, data.dataset)
    path = os.path.join(path, data.marker)
    res = 'h%s_w%s_n%s_zdim%s' % (data.patch_h, data.patch_w, data.n_channels, model.z_dim)
    path = os.path.join(path, res)
    img_path = os.path.join(path, 'generated_images')
    if not os.path.isdir(path):
        os.makedirs(path)
    if not os.path.isdir(img_path):
        os.makedirs(img_path)

    hdf5_path = os.path.join(path, 'hdf5_%s_%s_images_%s.h5' % (data.dataset, data.marker, model.model_name))
    
    # Lazy access to one set of images, not used at all, just filling tensorflows complains.
    ds_o = data.training
    if ds_o is None:
        ds_o = data.test
    if ds_o is None:
        ds_o = data.validation
    for batch_images, batch_labels in ds_o:
        break
    
    if not os.path.isfile(hdf5_path):
        # H5 File specifications and creation.
        img_shape = [num_samples] + data.test.shape[1:]
        latent_shape = [num_samples] + [model.z_dim]
        hdf5_file = h5py.File(hdf5_path, mode='w')
        img_storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)
        z_storage = hdf5_file.create_dataset(name='z_latent', shape=latent_shape, dtype=np.float32)
        if 'PathologyGAN' in model.model_name:
            w_storage = hdf5_file.create_dataset(name='w_latent', shape=latent_shape, dtype=np.float32)
        print('Generated Images path:', img_path)
        print('H5 File path:', hdf5_path)

        saver = tf.train.Saver()
        with tf.Session() as session:

            # Initializer and restoring model.
            session.run(tf.global_variables_initializer())
            saver.restore(session, checkpoint)

            ind = 0
            while ind < num_samples:
                
                # Image and latent generation for PathologyGAN.
                if model.model_name == 'BigGAN':
                    z_latent_batch = np.random.normal(size=(batches, model.z_dim))
                    feed_dict = {model.z_input:z_latent_batch, model.real_images:batch_images}
                    gen_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

                # Image and latent generation for StylePathologyGAN.
                else:
                    # if exemplars are given then we'll generate a batch made up of the different linear combinations of them
                    if exemplar1 is not None and exemplar2 is not None:
                        print('Generating image interpolations from exemplars')
                        interpolation_weights = np.linspace(0.0, 1.0, num=batches)
                        printable_alphas = {i:round(interpolation_weights[i], 2) for i in range(len(interpolation_weights))}
                        z_latent_batch = np.array([alpha*exemplar1 + (1-alpha)*exemplar2 for alpha in interpolation_weights])
                    else:
                        z_latent_batch = np.random.normal(size=(batches, model.z_dim))
                    # ------- saving ordered dict with latent codes for generated images -------
                    latents_dict = {i: z_latent_batch[i] for i in range(len(z_latent_batch))}
                    print(f'writing z_latent_batch to pickle at {img_path}')
                    with open(f'{img_path}/latents_dict_{ind}.pkl', 'wb') as handle:
                        pickle.dump(latents_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    # --------------------------------------------------------------------------
                    feed_dict = {model.z_input_1: z_latent_batch, model.real_images:batch_images}
                    w_latent_batch = session.run([model.w_latent_out], feed_dict=feed_dict)[0]
                    w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])
                    feed_dict = {model.w_latent_in:w_latent_in, model.real_images:batch_images}
                    gen_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

                # Fill in storage for latent and image.
                for i in range(batches):
                    if ind == num_samples:
                        break
                    img_storage[ind] = gen_img_batch[i, :, :, :]
                    z_storage[ind] = z_latent_batch[i, :]
                    if 'PathologyGAN' in model.model_name:
                        w_storage[ind] = w_latent_batch[i, :]
                    # going to add (10*) the alpha value to the image name for easy cross-reference
                    plt.imsave('%s/gen_%s_alpha_%s.png' % (img_path, ind, int(100*printable_alphas[ind])), gen_img_batch[i, :, :, :])
                    # plt.imsave('%s/gen_%s.png' % (img_path, ind), gen_img_batch[i, :, :, :])
                    ind += 1
        print(ind, 'Generated Images')
    else:
        print('H5 File already created.')
        print('H5 File Generated Samples')
        print('\tFile:', hdf5_path)

    return hdf5_path

# Generate and encode samples from PathologyGAN, with an encoder.
def generate_encode_samples_from_checkpoint(model, data, data_out_path, checkpoint, num_samples=5000, batches=50):
    path = os.path.join(data_out_path, 'evaluation')
    path = os.path.join(path, model.model_name)
    path = os.path.join(path, data.dataset)
    path = os.path.join(path, data.marker)
    res = 'h%s_w%s_n%s' % (data.patch_h, data.patch_w, data.n_channels)
    path = os.path.join(path, res)
    img_path = os.path.join(path, 'generated_images')
    if not os.path.isdir(path):
        os.makedirs(path)
    if not os.path.isdir(img_path):
        os.makedirs(img_path)

    hdf5_path = os.path.join(path, 'hdf5_%s_%s_images_%s.h5' % (data.dataset, data.marker, model.model_name))
    
    # Lazy access to one set of images, not used at all, just filling tensorflows complains.
    batch_images = np.ones((data.batch_size, data.patch_h, data.patch_w, data.n_channels))

    if not os.path.isfile(hdf5_path):
        # H5 File specifications and creation.
        img_shape = [num_samples] + [data.patch_h, data.patch_w, data.n_channels]
        latent_shape = [num_samples] + [model.z_dim]
        hdf5_file = h5py.File(hdf5_path, mode='w')
        z_storage = hdf5_file.create_dataset(name='z_latent', shape=latent_shape, dtype=np.float32)
        # Generated images.
        img_storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)
        w_storage = hdf5_file.create_dataset(name='w_latent', shape=latent_shape, dtype=np.float32)
        # Reconstructed generated images.
        img_prime_storage = hdf5_file.create_dataset(name='images_prime', shape=img_shape, dtype=np.float32)
        w_prime_storage = hdf5_file.create_dataset(name='w_latent_prime', shape=latent_shape, dtype=np.float32)
        print('Generated Images path:', img_path)
        print('H5 File path:', hdf5_path)

        saver = tf.train.Saver()
        with tf.Session() as session:

            # Initializer and restoring model.
            session.run(tf.global_variables_initializer())
            saver.restore(session, checkpoint)

            ind = 0
            while ind < num_samples:
                
                # W latent.
                z_latent_batch = np.random.normal(size=(batches, model.z_dim))
                feed_dict = {model.z_input_1: z_latent_batch}
                w_latent_batch = session.run([model.w_latent_out], feed_dict=feed_dict)[0]
                w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])

                # Generate images from W latent space.
                feed_dict = {model.w_latent_in:w_latent_in}
                gen_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

                # Encode generated images into W' latent space.
                feed_dict = {model.real_images_2:gen_img_batch}
                w_latent_prime_batch = session.run([model.w_latent_e_out], feed_dict=feed_dict)[0]
                w_latent_prime_in = np.tile(w_latent_prime_batch[:,:, np.newaxis], [1, 1, model.layers+1])

                # Generate images from W' latent space.
                feed_dict = {model.w_latent_in:w_latent_prime_in}
                gen_img_prime_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

                # Fill in storage for latent and image.
                for i in range(batches):
                    if ind == num_samples:
                        break
                    z_storage[ind] = z_latent_batch[i, :]
                    # Generated.
                    img_storage[ind] = gen_img_batch[i, :, :, :]
                    w_storage[ind] = w_latent_batch[i, :]
                    # Reconstructed.
                    img_prime_storage[ind] = gen_img_prime_batch[i, :, :, :]
                    w_prime_storage[ind] = w_latent_prime_batch[i, :]

                    # Saving images
                    plt.imsave('%s/gen_%s.png' % (img_path, ind), gen_img_batch[i, :, :, :])
                    plt.imsave('%s/gen_recon_%s.png' % (img_path, ind), gen_img_prime_batch[i, :, :, :])
                    ind += 1
        print(ind, 'Generated Images')
    else:
        print('H5 File already created.')
        print('H5 File Generated Samples')
        print('\tFile:', hdf5_path)

    return hdf5_path

# Encode real images and regenerate from PathologyGAN, with an encoder.
def real_encode_eval_from_checkpoint(model, data, data_out_path, checkpoint, real_hdf5, type_set, num_samples=5000, batches=50):
    path = os.path.join(data_out_path, 'evaluation')
    path = os.path.join(path, model.model_name)
    path = os.path.join(path, data.dataset)
    path = os.path.join(path, data.marker)
    res = 'h%s_w%s_n%s' % (data.patch_h, data.patch_w, data.n_channels)
    path = os.path.join(path, res)
    img_path = os.path.join(path, 'real_images')
    if not os.path.isdir(path):
        os.makedirs(path)
    if not os.path.isdir(img_path):
        os.makedirs(img_path)

    if not os.path.isfile(real_hdf5):
        print('Real image H5 file does not exist:', real_hdf5)
        exit()
    real_images = read_hdf5(real_hdf5, 'images')

    hdf5_path = os.path.join(path, 'hdf5_%s_%s_real_%s_images_%s.h5' % (data.dataset, data.marker, type_set, model.model_name))
    
    # Lazy access to one set of images, not used at all, just filling tensorflows complains.
    batch_images = np.ones((data.batch_size, data.patch_h, data.patch_w, data.n_channels))

    if not os.path.isfile(hdf5_path):
        # H5 File specifications and creation.
        img_shape = [num_samples] + [data.patch_h, data.patch_w, data.n_channels]
        latent_shape = [num_samples] + [model.z_dim]
        hdf5_file = h5py.File(hdf5_path, mode='w')
        # Real images.
        img_storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)
        w_storage = hdf5_file.create_dataset(name='w_latent', shape=latent_shape, dtype=np.float32)
        # Reconstructed generated images.
        img_prime_storage = hdf5_file.create_dataset(name='images_prime', shape=img_shape, dtype=np.float32)
        w_prime_storage = hdf5_file.create_dataset(name='w_latent_prime', shape=latent_shape, dtype=np.float32)
        print('Generated Images path:', img_path)
        print('H5 File path:', hdf5_path)

        saver = tf.train.Saver()
        with tf.Session() as session:

            # Initializer and restoring model.
            session.run(tf.global_variables_initializer())
            saver.restore(session, checkpoint)

            ind = 0
            while ind < num_samples:

                # Real images.
                if (ind + batches) < len(real_images):
                    real_img_batch = real_images[ind: ind+batches, :, :, :]/255.
                else:
                    real_img_batch = real_images[ind:, :, :, :]/255.

                # Encode real images into W latent space.
                feed_dict = {model.real_images_2:real_img_batch}
                w_latent_batch = session.run([model.w_latent_e_out], feed_dict=feed_dict)[0]
                w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])

                # Generate images from W latent space.
                feed_dict = {model.w_latent_in:w_latent_in}
                recon_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

                # Encode reconstructed images into W' latent space.
                feed_dict = {model.real_images_2:recon_img_batch}
                w_latent_prime_batch = session.run([model.w_latent_e_out], feed_dict=feed_dict)[0]
                w_latent_prime_in = np.tile(w_latent_prime_batch[:,:, np.newaxis], [1, 1, model.layers+1])

                # Fill in storage for latent and image.
                for i in range(batches):
                    if ind == num_samples:
                        break
                    # Real Images.
                    img_storage[ind] = real_img_batch[i, :, :, :]
                    w_storage[ind] = w_latent_batch[i, :]
                    
                    # Reconstructed images.
                    img_prime_storage[ind] = recon_img_batch[i, :, :, :]
                    w_prime_storage[ind] = w_latent_prime_batch[i, :]

                    # Saving images
                    plt.imsave('%s/real_%s.png' % (img_path, ind), real_img_batch[i, :, :, :])
                    plt.imsave('%s/real_recon_%s.png' % (img_path, ind), recon_img_batch[i, :, :, :])
                    ind += 1
        print(ind, 'Generated Images')
    else:
        print('H5 File already created.')
        print('H5 File Generated Samples')
        print('\tFile:', hdf5_path)

    return hdf5_path

# Encode real images for prognosis.
def real_encode_from_checkpoint(model, data, data_out_path, checkpoint, real_hdf5, batches=50, save_img=False):
    os.umask(0o002)
    path = os.path.join(data_out_path, 'evaluation')
    path = os.path.join(path, model.model_name)
    path = os.path.join(path, data.dataset)
    # path = os.path.join(path, data.marker)
    res = 'h%s_w%s_n%s_zdim%s' % (data.patch_h, data.patch_w, data.n_channels, model.z_dim)
    path = os.path.join(path, res)
    img_path = os.path.join(path, 'real_images_recon')
    if not os.path.isdir(path):
        os.makedirs(path)
    if save_img and not os.path.isdir(img_path):
        os.makedirs(img_path)

    if not os.path.isfile(real_hdf5):
        print('Real image H5 file does not exist:', real_hdf5)
        exit()

    real_images = read_hdf5(real_hdf5, 'images')
    real_labels = read_hdf5(real_hdf5, 'labels')
    real_names = read_hdf5(real_hdf5, 'file_name')
    num_samples = real_images.shape[0]

    name_file = real_hdf5.split('/')[-1]
    hdf5_path = os.path.join(path, name_file)

    # Lazy access to one set of images, not used at all, just filling tensorflows complains.
    batch_images = np.ones((data.batch_size, data.patch_h, data.patch_w, data.n_channels))
    
    if not os.path.isfile(hdf5_path):
        # H5 File specifications and creation.
        img_shape = real_images.shape
        labels_shape = real_labels.shape
        latent_shape = [num_samples] + [model.z_dim]
        with h5py.File(hdf5_path, mode='w') as hdf5_file:
        
            # Reconstructed generated images.
            img_storage = hdf5_file.create_dataset(name='images_prime', shape=img_shape, dtype=np.float32)
            labels_storage = hdf5_file.create_dataset(name='labels', shape=labels_shape, dtype=np.float32)
            w_storage = hdf5_file.create_dataset(name='w_latent', shape=latent_shape, dtype=np.float32)
            if real_names is not None:
                dt = h5py.special_dtype(vlen=str)
                names_storage = hdf5_file.create_dataset(name='file_name', shape=(num_samples,1), dtype=dt)

            print('Generated Images path:', img_path)
            print('H5 File path:', hdf5_path)

            saver = tf.train.Saver()
            with tf.Session() as session:

                # Initializer and restoring model.
                session.run(tf.global_variables_initializer())
                saver.restore(session, checkpoint)

                print('Number of Real Images:', num_samples)
                print('Starting encoding...')
                ind = 0
                while ind < num_samples:

                    # Real images.
                    if (ind + batches) < len(real_images):
                        real_img_batch = real_images[ind: ind+batches, :, :, :]/255.
                        real_labels_batch = real_labels[ind: ind+batches, :]
                        if real_names is not None: real_names_batch = real_names[ind: ind+batches, :]
                        
                    else:
                        real_img_batch = real_images[ind:, :, :, :]/255.
                        real_labels_batch = real_labels[ind:, :]
                        if real_names is not None: real_names_batch = real_names[ind:, :]

                    # Encode real images into W latent space.
                    feed_dict = {model.real_images_2:real_img_batch}
                    w_latent_batch = session.run([model.w_latent_e_out], feed_dict=feed_dict)[0]
                    w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])

                    # Generate images from W latent space.
                    feed_dict = {model.w_latent_in:w_latent_in}
                    recon_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

                    # Fill in storage for latent and image.
                    for i in range(batches):
                        if ind == num_samples:
                            break

                        # Reconstructed images.
                        img_storage[ind] = recon_img_batch[i, :, :, :]
                        labels_storage[ind] = real_labels_batch[i, :]
                        if real_names is not None: names_storage[ind] = real_names_batch[i, :]
                        w_storage[ind] = w_latent_batch[i, :]

                        if save_img:
                            # Saving images
                            plt.imsave('%s/real_recon_%s.png' % (img_path, ind), recon_img_batch[i, :, :, :])
                        ind += 1

                    if ind%10000==0: print('Processed', ind, 'images')

        print(ind, 'Encoded Images')
    else:
        print('H5 File already created.')
        print('H5 File Generated Samples')
        print('\tFile:', hdf5_path)

    return hdf5_path, num_samples

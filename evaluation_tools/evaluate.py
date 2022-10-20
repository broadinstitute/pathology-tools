import argparse
import h5py
import numpy as np
import pickle
from FID.fid import get_fid


def FID_dataset_prep(synth_dataset, training_dataset, n_samples, output_dir=None, save_datasets=True):
    # data preparation – input: paths to image datasets the were input/output
    # for PathologyGAN, number of samples to be drawn for FID calculation
    # --> output: will write the real/fake datasets to pickle files at specified path
    synth_imgs = h5py.File(synth_dataset, 'r')
    training_imgs = h5py.File(training_dataset, 'r')
    synth_arr = np.array(synth_imgs['images'])
    training_arr = np.array(training_imgs['images'])

    # select min(n_samples, len(dataset)) random draws from the datasets
    # TODO: Question -- does the FID calculation require the same numbers of each
    #  dataset to be given? Intuitively that shouldn't be necessary but not sure about the implementation
    print(f'synth_arr.shape = {synth_arr.shape}')
    print(f'training_arr.shape = {training_arr.shape}')
    if n_samples < len(synth_arr):
        synth_inds = [np.random.choice(len(synth_arr), n_samples, replace=False)]
        synth_arr = synth_arr[synth_inds]
    if n_samples < len(training_arr):
        training_inds = [np.random.choice(len(training_arr), n_samples, replace=False)]
        training_arr = training_arr[training_inds]

    print(f'synth_arr.shape = {synth_arr.shape}')
    print(f'training_arr.shape = {training_arr.shape}')

    # scaling the images to the right range of pixel values (**assuming PathologyGAN output with pixels \in (0,1))
    synth_arr = np.multiply(synth_arr, 255).astype(np.uint8)
    training_arr = np.multiply(training_arr, 255).astype(np.uint8)
    synth_arr = np.reshape(synth_arr, (len(synth_arr), 3, 224, 224))
    training_arr = np.reshape(training_arr, (len(training_arr), 3, 224, 224))

    if save_datasets:
        assert output_dir is not None, 'To save datasets to pickle, must provide an output directory path'
        with open(output_dir + 'synth_FID_samples.pkl', 'wb') as f:
            pickle.dump(synth_arr, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(output_dir + 'training_FID_samples.pkl', 'wb') as f:
            pickle.dump(training_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    return synth_arr, training_arr


parser = argparse.ArgumentParser(description='PathologGAN FID/IS evaluation')
parser.add_argument('--synth_dataset', help='Path to synthetic (PathologyGAN output) dataset')
parser.add_argument('--training_dataset', help='Path to training dataset used for PathologyGAN')
parser.add_argument('--n_samples_FID', type=int, default=5000, help='Number of samples considered in FID calculation')
parser.add_argument('--pickle_output_dir', default=None, help='Output directory for dataset pickles to be written')
parser.add_argument('--save_FID_datasets', type=bool, default=True, help='Bool flag to trigger saving of FID datasets')
args = parser.parse_args()

# checking file types and ensuring valid path
assert args.synth_dataset[-3:] == '.h5' and args.training_dataset[-3:] == '.h5', 'Datasets must be .h5 files'
if args.pickle_output_dir is not None and args.pickle_output_dir[-1] != '/':
    args.pickle_output_dir += '/'

synth_FID_dataset, real_FID_dataset = FID_dataset_prep(args.synth_dataset, args.training_dataset, args.n_samples_FID,
                                                       output_dir=args.pickle_output_dir,
                                                       save_datasets=args.save_FID_datasets)
print(f'FID score on {args.n_samples_FID} samples = {get_fid(synth_FID_dataset, real_FID_dataset)}')

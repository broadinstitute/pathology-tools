import argparse
import h5py
import numpy as np
import pickle
from FID.fid import get_fid
from IS.inception_score import get_inception_score


def dataset_prep(synth_dataset, n_samples, training_dataset=None, output_dir=None, save_datasets=True, metric=None):
    # data preparation – input: paths to image datasets the were input/output
    # for PathologyGAN, number of samples to be drawn for FID or IS calculation
    # --> output: will write the real/fake datasets to pickle files at specified path
    assert metric.upper() in ['FID', 'IS'], 'Need to provide \'FID\' or \'IS\' as input for parameter \'metric\''
    if metric == 'FID':
        assert training_dataset is not None, 'If calculating FID, must provide path to training set .h5 file'

    synth_imgs = h5py.File(synth_dataset, 'r')
    synth_arr = np.array(synth_imgs['images'])

    # select min(n_samples, len(dataset)) random draws from the datasets
    # TODO: Question -- does the FID calculation require the same numbers of each
    #  dataset to be given? Intuitively that shouldn't be necessary but not sure about the implementation
    if n_samples < len(synth_arr):
        synth_inds = [np.random.choice(len(synth_arr), n_samples, replace=False)]
        synth_arr = np.squeeze(synth_arr[synth_inds])  # <- this appears to introduce an extra dimension of size 1...

    # scaling the images to the right range of pixel values (**assuming PathologyGAN output with pixels \in (0,1))
    synth_arr = np.multiply(synth_arr, 255).astype(np.uint8)
    synth_arr = np.reshape(synth_arr, (len(synth_arr), 3, 224, 224))

    # If calculating FID, also generate the dataset of real images
    if metric == 'FID':
        training_imgs = h5py.File(training_dataset, 'r')
        training_arr = np.array(training_imgs['images'])
        if n_samples < len(training_arr):
            training_inds = [np.random.choice(len(training_arr), n_samples, replace=False)]
            training_arr = np.squeeze(training_arr[training_inds])
        training_arr = np.multiply(training_arr, 255).astype(np.uint8)
        training_arr = np.reshape(training_arr, (len(training_arr), 3, 224, 224))

    if save_datasets:
        assert output_dir is not None, 'To save datasets to pickle, must provide an output directory path'
        with open(output_dir + 'synth_FID_samples.pkl', 'wb') as f:
            pickle.dump(synth_arr, f, protocol=pickle.HIGHEST_PROTOCOL)
        if metric == 'FID':
            with open(output_dir + 'training_FID_samples.pkl', 'wb') as f:
                pickle.dump(training_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    if metric == 'FID':
        return synth_arr, training_arr
    else:
        return synth_arr


parser = argparse.ArgumentParser(description='PathologGAN FID/IS evaluation')
parser.add_argument('--FID', action='store_true', help='Bool flag to trigger FID calculation')
parser.add_argument('--IS', action='store_true', help='Bool flag to trigger IS calculation')
parser.add_argument('--synth_dataset', help='Path to synthetic (PathologyGAN output) dataset')
parser.add_argument('--training_dataset', default=None, help='Path to training dataset used for PathologyGAN')
parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples considered in metric calculation')
parser.add_argument('--pickle_output_dir', default=None, help='Output directory for dataset pickles to be written')
parser.add_argument('--save_datasets', action='store_true', help='Bool flag to trigger saving of datasets')
args = parser.parse_args()

# checking file types and ensuring valid path
assert args.synth_dataset[-3:] == '.h5' and (args.training_dataset is None or args.training_dataset[-3:] == '.h5'),\
    'Datasets must be .h5 files'
if args.pickle_output_dir is not None and args.pickle_output_dir[-1] != '/':
    args.pickle_output_dir += '/'

print(f'METRICS BEING CALCULATED -- FID:{args.FID}; IS:{args.IS}')
assert args.FID or args.IS, 'Must provide at CLI flag --FID or --IS to calculate desired metric(s)'
print(f'args.save_datasets = {args.save_datasets}')

datasets = dataset_prep(args.synth_dataset, args.n_samples,
                        training_dataset=args.training_dataset,
                        output_dir=args.pickle_output_dir,
                        save_datasets=args.save_datasets,
                        metric=('FID' if args.FID else 'IS'))

if args.FID:
    print(f'FID score on {args.n_samples} samples = {get_fid(datasets[0], datasets[1])}')
if args.IS:
    print(f'Inception score on {args.n_samples} samples = {get_inception_score(datasets[0])}')

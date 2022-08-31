import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import random
import openslide
import histomicstk as htk
import cv2
import argparse


class BLCA_CL_Dataset(object):
    def __init__(self, path, mode='Train', train_prop=0.8, transform=None, return_PIL=False, resize_dim=None,
                 shuffle=False, stain_color_map=None):
        self.root = path
        self.data_transformation = transform
        # Flag for returning images in (width, height, channel) shape with pixels
        # in 0-255 --> this is what you get from casting a PIL image to a numpy array, and its
        # what PathologyGAN expects for its input images
        self.return_PIL_format = return_PIL
        # ... and an optional slot to give a resize dimension (default PathologyGAN expects 224x224 images for example)
        self.resize_dim = resize_dim
        # optional stain color map in case we're filtering out patches with green marker
        self.stain_color_map = stain_color_map
        # boolean flag for whether or not call into green filtering method
        self.identify_green = self.stain_color_map is not None
        print(f'Dataset attribute self.identify_green = {self.identify_green}')

        files = os.listdir(self.root)
        h5s = [x for x in files if '.h5' in x]  # <- whole-slide images
        # *OPTIONAL* Shuffling the list of patients before splitting on patient -- contents of the two splits
        # determined by the seed and the split proportion
        random.seed(4)
        if shuffle:
            random.shuffle(h5s)
        print(f'h5s = {h5s}')
        # patient labels from the .h5 files -- given by the first three substrings in the filename
        h5s_pat = sorted(list(set(['-'.join(x.split('-')[:3]) for x in h5s])))

        # make the train/val split at the patient level (necessary in the general case of having \geq 1 samples per patient)
        if mode == 'Train':
            h5s_pat = h5s_pat[:int(train_prop * len(h5s_pat))]
        else:  # <-- assuming only two splits, just train and val
            h5s_pat = h5s_pat[int(train_prop * len(h5s_pat)):]

        # Select the WSIs for which the patient label is found in the set of approved patient labels (h5s_pat)
        # final_list = [ele for ele in h5s if any([pref in ele for pref in h5s_pat])]
        final_list = h5s
        print(f'Number of .h5 files being included in {mode} dataset = {len(final_list)}')

        self.coords_all = []

        # for j in tqdm(final_list):
        for j in final_list:
            with h5py.File(self.root + j, "r") as f:
                # List all groups
                # print("Keys: %s" % f.keys())
                a_group_key = list(f.keys())[0]

                patch_level = f['coords'].attrs['patch_level']
                patch_size = f['coords'].attrs['patch_size']

                # Get the data (the coordinates for each patch's top left corner?)
                data = list(f[a_group_key])

                for k in data:
                    self.coords_all.append([j, k, patch_level, patch_size])

    def __getitem__(self, idx):
        transform = self.data_transformation
        current_patch = self.coords_all[idx]
        slide = openslide.OpenSlide(self.root + current_patch[0][:-3] + '.svs')
        img = slide.read_region(tuple(current_patch[1]), current_patch[2],
                                tuple([current_patch[3], current_patch[3]])).convert('RGB')
        # if we want images in the format required by PathologyGAN, we just cast the PIL RGB images to numpy arrays
        if self.return_PIL_format:
            if self.resize_dim is not None:
                # PIL.Image has resize functions, ANTIALIAS is supposed to be best for scaling down
                img = img.resize((self.resize_dim, self.resize_dim), Image.ANTIALIAS)
                img = np.array(img)
            if self.identify_green:
                # if we want to separate patches with marker, pass the PIL image to filter_green which returns
                # the numpy array and a bool flag indicating whether the image has marker
                return self.filter_green(img, self.stain_color_map, img_size=self.resize_dim)
            else:
                return img

        # if the dataset is instantiated with a transform function then we'll use it, otherwise we create on just consisting of ToTensor
        if not transform:
            transform = transforms.Compose([transforms.ToTensor()])
            return transform(img)
        else:
            return transform(img)

    def get_specific_item(self, slide_path, x, y, resize_dim):
        # helper method to generate patch images from slide/coord inputs -- being used to check the patches
        # identified as green by the detect_green method
        patch_coords = np.array([int(x), int(y)])
        slide = openslide.OpenSlide(self.root + slide_path[:-3] + '.svs')
        img = slide.read_region(tuple(patch_coords), 0, tuple([256, 256])).convert('RGB')
        if self.return_PIL_format:
            if self.resize_dim is not None:
                img = img.resize((resize_dim, resize_dim), Image.ANTIALIAS)
            return np.array(img)

    def filter_green(self, img, stain_color_map, img_size=488):
        # function to check for presence of marker using histomicstk color deconvolution
        # input: PIL img (output of img.resize() command in __getitem__())
        # --> usage: will return numpy array of image along with Bool flag indicating whether or not
        # ---> the patch contains marker. When including this filtering step, separate hdf5 dataset files will
        # ---> be accumulated for green and non-green images
        stains = list(stain_color_map.keys())
        w_est = np.array([stain_color_map[st] for st in stains]).T
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # color deconv
        deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(img_rgb, w_est, 255)
        # ----- tighten this up -----
        green_marker = deconv_result.Stains[:, :, 1]
        tissue = deconv_result.Stains[:, :, 0]
        green_marker[green_marker > 150] = 0
        green_marker[green_marker > 0] = 1
        tissue[tissue > 200] = 0
        tissue[tissue > 0] = 1
        # ---------------------------
        contours, hierarchy = cv2.findContours(green_marker.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        max_contour = 0
        for contour in contours:
            contour_size = cv2.contourArea(contour)
            if contour_size > max_contour:
                max_contour = contour_size

        if max_contour > 10000 or np.sum(tissue) < (img_size * img_size * 0.1):
            # green
            return np.array(img), True
        else:
            # not green
            return np.array(img), False

    def __len__(self):
        # Length of dataset given by number of overall number of patches across all slides
        return len(self.coords_all)


def construct_hdf5_datasets(input_patches_dir, output_prefix, train_prop=1.0, img_dim=224, max_dataset_size=None,
                            shuffle=False, stain_color_map=None):
    # function to create hdf5 files containing training and testing image datasets
    # -> Intended to create datasets files in format required by PathologyGAN training procedure
    filter_green = stain_color_map is not None
    print(f'construct_hdf5_dataset() called with filter_green={filter_green}')

    # generate dataset objects that return numpy array images in the format and size required by PathologyGAN
    train_dataset = BLCA_CL_Dataset(input_patches_dir, train_prop=train_prop,
                                    mode='Train', return_PIL=True, resize_dim=img_dim, shuffle=shuffle,
                                    stain_color_map=stain_color_map)

    # initialize and populate lists of images
    train_list = []
    # if we're filtering green patches out, we'll put the non-green images in train_list, and separately
    # accumulate the train_list_green list to keep track of the green images
    train_list_green = []

    # impose optional maximum dataset size (to allow for small dataset sizes when experimenting)
    if max_dataset_size:
        trainset_size = min(len(train_dataset), max_dataset_size)
    else:
        trainset_size = len(train_dataset)

    print(f'Training set size = {trainset_size}')

    if not filter_green:
        for i in tqdm(range(trainset_size)):
            train_list.append(train_dataset.__getitem__(i))
    else:
        i = 0
        while len(train_list) < trainset_size:
            img, green = train_dataset.__getitem__(i)
            print(f'Image #{i} from dataset is {"not" if green else ""} green')
            if green:
                train_list_green.append(img)
            else:
                train_list.append(img)
            i += 1

    # save datasets to hdf5
    with h5py.File(output_prefix + '_train.h5', 'w') as f:
        # train_dset = f.create_dataset('images', data=np.array(train_list))
        f.create_dataset('images', data=np.array(train_list), compression='gzip')
        f.close()

    # ... and optionally the green dataset
    with h5py.File(output_prefix + '_train_GREEN.h5', 'w') as f:
        f.create_dataset('images', data=np.array(train_list), compression='gzip')
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Histology patch dataset generator - currently tailored to be used'
                                                 'for generating .h5 dataset files for use by PathologyGAN')
    parser.add_argument('--input_patches_dir', type=str, help='Directory containing CLAM-generated patch files (h5s and'
                                                              'svs files')
    parser.add_argument('--output_prefix', type=str, help='Output prefix for training .h5 dataset file')
    parser.add_argument('--train_proportion', type=float, default=1.0, help='Proportion of data to use for training'
                                                                            ' set in rain/test split')
    parser.add_argument('--img_dim', type=int, default=448, help='Dimension (side-length) for output images')
    parser.add_argument('--max_dataset_size', type=int, help='Maximum number of samples to include in .h5 dataset')
    parser.add_argument('--shuffle', type=bool, default=False, help='Boolean indicating whether or not to shuffle the '
                                                                    'hdf5 files being used to build the dataset')
    args = parser.parse_args()

    # defining the stain color map used in the green filtering step
    green_filter_cmap = {'tissue': [0.24334306, 0.79350096, 0.55779959],
                         'green_marker': [0.90135287, 0.26190666, 0.34491723],
                         'other': [0.16255852, 0.5335877, -0.82997523]}

    # --- setting the main method to generate hdf5 datasets in format for pathology-gan training ---
    # example output prefix: '/workdir/crohlice/scripts/PurityGAN/Pathology-GAN/dataset/tcga/he/patches_h448_w448/TESTLARGE_hdf5_tcga_he',
    construct_hdf5_datasets(input_patches_dir=args.input_patches_dir, output_prefix=args.output_prefix,
                            train_prop=args.train_proportion,
                            img_dim=args.img_dim, max_dataset_size=args.max_dataset_size, shuffle=args.shuffle,
                            stain_color_map=green_filter_cmap)

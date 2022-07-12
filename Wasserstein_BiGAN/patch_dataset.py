# Modified version of Matt's dataset/dataloader script
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

class BLCA_CL_Dataset(object):
    def __init__(self, path, mode='Train', train_prop=0.8, transform=None, return_PIL=False, resize_dim=None,
                 shuffle=False):
        self.root = path
        self.data_transformation = transform
        # Flag for returning images in (width, height, channel) shape with pixels
        # in 0-255 --> this is what you get from casting a PIL image to a numpy array, and its
        # what PathologyGAN expects for its input images
        self.return_PIL_format = return_PIL
        # ... and an optional slot to give a resize dimension (default PathologyGAN expects 224x224 images for example)
        self.resize_dim = resize_dim

        files = os.listdir(self.root)
        h5s = [x for x in files if '.h5' in x] #<- whole-slide images
        # patient labels from the .h5 files -- given by the first three substrings in the filename
        h5s_pat = sorted(list(set(['-'.join(x.split('-')[:3]) for x in h5s])))
        # *OPTIONAL* Shuffling the list of patients before splitting on patient -- contents of the two splits
        # determined by the seed and the split proportion
        random.seed(4)
        if shuffle:
            random.shuffle(h5s_pat)
        #print(h5s_pat)

        # make the train/val split at the patient level (necessary in the general case of having \geq 1 samples per patient)
        if mode == 'Train':
            h5s_pat = h5s_pat[:int(train_prop*len(h5s_pat))]
        else: #<-- assuming only two splits, just train and val
            h5s_pat = h5s_pat[int(train_prop*len(h5s_pat)):]

        # Select the WSIs for which the patient label is found in the set of approved patient labels (h5s_pat)
        # final_list = [ele for ele in h5s if any([pref in ele for pref in h5s_pat])]
        final_list = h5s
        print(f'Number of .h5 files being included in {mode} dataset = {len(final_list)}')

        self.coords_all = []

        # for j in tqdm(final_list):
        for j in final_list:
            with h5py.File(self.root+j, "r") as f:
                # List all groups
                #print("Keys: %s" % f.keys())
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
            slide = openslide.OpenSlide(self.root+current_patch[0][:-3]+'.svs')
            img = slide.read_region(tuple(current_patch[1]), current_patch[2], tuple([current_patch[3], current_patch[3]])).convert('RGB')
            # if we want images in the format required by PathologyGAN, we just cast the PIL RGB images to numpy arrays
            if self.return_PIL_format:
                if self.resize_dim is not None:
                    # PIL.Image has resize functions, ANTIALIAS is supposed to be best for scaling down
                    img = img.resize((self.resize_dim, self.resize_dim), Image.ANTIALIAS)
                    # *** Adding print statements revealing the source slide and current_patch for patches whose green channel
                    # has an average value > 200 (over a random 5000 sample, the avg/std is 183/29)
                    np_img = np.array(img)
                    if np.mean(np_img[:, :, 1]) > 200:
                        print(f'{slide}\t{current_patch}')
                return np_img #np.array(img)

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

    def __len__(self):
        # Length of dataset given by number of overall number of patches across all slides
        return len(self.coords_all)

def detect_green():
    # function to identify slide and patch coordinates of TCGA patches appearing green
    dataset = BLCA_CL_Dataset('/workdir/crohlice/software/CLAM/TCGA_svs_h5_256/', train_prop=1.0, mode='Train',
                              return_PIL=True, resize_dim=224)
    # iterate over dataset to accumulate the print statements identifying green patches
    for i in range(len(dataset)):
        dataset.__getitem__(i)

def generate_green_patches(patch_csv, output_dim, output_file):
    # method to parse csv with lines in the format "slide_path, x_coord, y_coord"
    # and output a pickle file with the numpy array images
    dataset = BLCA_CL_Dataset('/workdir/crohlice/software/CLAM/TCGA_svs_h5_256/', train_prop=1.0, mode='Train',
                              return_PIL=True, resize_dim=output_dim)
    patch_list = []
    with open(patch_csv) as f:
        for line in tqdm(f):
            ln = line.split(',')
            patch_list.append(dataset.get_specific_item(ln[0], ln[1], ln[2], output_dim))
    # write output file to numpy binary .npy file
    with open(output_file, 'wb') as g:
        np.save(g, np.array(patch_list))

def construct_hdf5_datasets(output_prefix, train_prop=0.8, img_dim=224, max_dataset_size=10000):
    # function to create hdf5 files containing training and testing image datasets
    # -> Intended to create datasets files in format required by PathologyGAN training procedure

    # generate dataset objects that return numpy array images in the format and size required by PathologyGAN
    train_dataset = BLCA_CL_Dataset('/workdir/crohlice/software/CLAM/TCGA_svs_h5_256/', train_prop=train_prop,
                                    mode='Train', return_PIL=True, resize_dim=img_dim)
    # test_dataset = BLCA_CL_Dataset('/workdir/crohlice/software/CLAM/TCGA_svs_h5_256/', train_prop=train_prop,
    #                                 mode='Test', return_PIL=True, resize_dim=img_dim)

    # initialize and populate lists of images
    train_list = []
    # test_list = []
    # impose optional maximum dataset size (to allow for small dataset sizes when experimenting)
    if max_dataset_size:
        trainset_size = min(len(train_dataset), max_dataset_size)
        # testset_size = min(len(test_dataset), max_dataset_size)
    else:
        trainset_size = len(train_dataset)
        # testset_size = len(test_dataset)

    print(f'Training set size = {trainset_size}')

    for i in tqdm(range(trainset_size)):
        train_list.append(train_dataset.__getitem__(i))
    # for i in range(testset_size):
    #     test_list.append(test_dataset.__getitem__(i))

    # save datasets to hdf5
    with h5py.File(output_prefix+'_train.h5', 'w') as f:
        # train_dset = f.create_dataset('images', data=np.array(train_list))
        f.create_dataset('images', data=np.array(train_list))
        f.close()
    # checking difference in hdf5 file size with and without chunking
    with h5py.File(output_prefix + '_train_chunked.h5', 'w') as f:
        f.create_dataset('images', data=np.array(train_list), chunks=True)
        f.close()


    # with h5py.File(output_prefix + '_test.h5', 'w') as f:
    #     test_dset = f.create_dataset('images', data=np.array(test_list))
    #     f.close()

if __name__=='__main__':
    # --- generating patches identified as green by looking for patches with avg green value > 200 ---
    generate_green_patches('slide_green_patch_coords.csv', 16, 'green_patches.npy')

    # --- setting the main method to generate hdf5 datasets in format for pathology-gan training ---
    # construct_hdf5_datasets('/workdir/crohlice/scripts/PurityGAN/Pathology-GAN/dataset/tcga/he/patches_h224_w224/hdf5_compression_test',
    #                         train_prop=1.0)
    # ----------------------------------------------------------------------------------------------
    # seed = 1234
    # pl.seed_everything(seed)
    # BATCH_SIZE=64

    # patch_dataset_train = BLCA_CL_Dataset('/workdir/crohlice/software/CLAM/TCGA_svs_h5_128/', mode='Train') ### put this as the folder with H5 files
    # patch_dataloader_train = DataLoader(patch_dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    # patch_dataset_val = BLCA_CL_Dataset('/workdir/crohlice/software/CLAM/TCGA_svs_h5_128/', mode='Val')
    # patch_dataloader_val = DataLoader(patch_dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    #torch.save(patch_dataloader_train, './ali-pytorch/patch_dataloaders/patch_dataloader_train.pt')
    #torch.save(patch_dataloader_val, './ali-pytorch/patch_dataloaders/patch_dataloader_val.pt')

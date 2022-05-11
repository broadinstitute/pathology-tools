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
    def __init__(self, path, mode='Train', train_prop=0.8, transform=None):
        self.root = path
        self.data_transformation = transform
        files = os.listdir(self.root)
        h5s = [x for x in files if '.h5' in x] #<- whole-slide images
        # patient labels from the .h5 files -- given by the first three substrings in the filename
        h5s_pat = sorted(list(set(['-'.join(x.split('-')[:3]) for x in h5s])))
        # Shuffling the list of patients before splitting on patient -- contents of the two splits
        # determined by the seed and the split proportion
        random.seed(4)
        random.shuffle(h5s_pat)
        #print(h5s_pat)

        # make the train/val split at the patient level (necessary in the general case of having \geq 1 samples per patient)
        if mode == 'Train':
            h5s_pat = h5s_pat[:int(train_prop*len(h5s_pat))]
        else: #<-- assuming only two splits, just train and val
            h5s_pat = h5s_pat[int(train_prop*len(h5s_pat)):]

        # I think this is the list the above code was trying to construct?
        final_list = [ele for ele in h5s if any([pref in ele for pref in h5s_pat])]
        #print(final_list)

        self.coords_all = []

        for j in tqdm(final_list):
            with h5py.File(self.root+j, "r") as f:
                # List all groups
                #print("Keys: %s" % f.keys())
                a_group_key = list(f.keys())[0]

                patch_level = f['coords'].attrs['patch_level']
                patch_size = f['coords'].attrs['patch_size']

                # Get the data (the coordinates for each patch's top left corner?)
                data = list(f[a_group_key])

                for k in data:
                    self.coords_all.append([j,k, patch_level, patch_size])

    def __getitem__(self, idx):
            transform = self.data_transformation
            current_patch = self.coords_all[idx]
            slide = openslide.OpenSlide(self.root+current_patch[0][:-3]+'.svs')
            img = slide.read_region(tuple(current_patch[1]), current_patch[2], tuple([current_patch[3], current_patch[3]])).convert('RGB')
            # if the dataset is instantiated with a transform function then we'll use it, otherwise we create on just consisting of ToTensor
            if not transform:
                transform = transforms.Compose([transforms.ToTensor()])
                return transform(img)
            else:
                return transform(img)
        
    def __len__(self):
        # Length of dataset given by number of overall number of patches across all slides
        return len(self.coords_all)

if __name__=='__main__':
    seed = 1234
    pl.seed_everything(seed)
    BATCH_SIZE=64
    
    patch_dataset_train = BLCA_CL_Dataset('/workdir/crohlice/software/CLAM/TCGA_svs_h5_128/', mode='Train') ### put this as the folder with H5 files
    patch_dataloader_train = DataLoader(patch_dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    patch_dataset_val = BLCA_CL_Dataset('/workdir/crohlice/software/CLAM/TCGA_svs_h5_128/', mode='Val')
    patch_dataloader_val = DataLoader(patch_dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    #torch.save(patch_dataloader_train, './ali-pytorch/patch_dataloaders/patch_dataloader_train.pt')
    #torch.save(patch_dataloader_val, './ali-pytorch/patch_dataloaders/patch_dataloader_val.pt')

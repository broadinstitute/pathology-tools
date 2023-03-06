import os

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np
import pandas
import h5py
from tqdm import tqdm

import lightly.embedding as embedding
import lightly.models as models
import lightly.loss as loss
import random

seed = 1234
pl.seed_everything(seed)

import openslide

import rawpy
#import imageio

# adapted by Rachel Honigsberg from Matt B


num_workers = 8
batch_size = 256
#memory_bank_size = 4096
#max_epochs = 20000000


class TCGA_Dataset(object):

    def __init__(self, path):

        self.root = path
#make sure this directory has both svs and h5 files and that it ends with "/" for file directory later on
        files = os.listdir("/workdir/rlh353/synthetic_dir/TCGA_svs_h5/TCGA_brca_all_WSI_full_256/")
        h5s = [x for x in files if '.h5' in x]

#        h5s_pat = sorted(list(set([x.split('-')[0] for x in h5s])))

 #       print(h5s_pat)
 #       random.seed(4)
 #       random.shuffle(h5s_pat)
 #       print(h5s_pat)

  #      pats_train = h5s_pat  # [:int(0.7*len(h5s_pat))]
  #      print(len(pats_train))

   #     final_list = {pref: ele for pref in pats_train for ele in h5s if pref in ele}
    #    final_list = list(final_list.values())
	
     #   print("final list \n")
    #    print(final_list)
        # np.save("/athena/ihlab/scratch/mbb4001/Bladder/train_list.npy", final_list)
        # np.save("E:\\Faltas\\train_list.npy", np.array(final_list))

        self.coords_all = []

        for j in tqdm(h5s):
            with h5py.File(self.root + j, "r") as f:
                # List all groups
                print("Keys: %s" % f.keys())
                a_group_key = list(f.keys())[0]

                patch_level = f['coords'].attrs['patch_level']
                patch_size = f['coords'].attrs['patch_size']

                print(patch_level, patch_size)

                # Get the data
                data = list(f[a_group_key])

                for k in data:
                    self.coords_all.append([j, k, patch_level, patch_size])

        print(self.coords_all[100:105], self.coords_all[10:25])

        current_patch = self.coords_all[50]
        slide = openslide.OpenSlide(self.root + current_patch[0][:-3] + '.svs')
        # print(tuple(current_patch[1]), current_patch[2], tuple([current_patch[3], current_patch[3]]))

        img = slide.read_region(tuple(current_patch[1]), current_patch[2],
                                tuple([current_patch[3], current_patch[3]])).convert('RGB')

#        print("comment test")

    def __getitem__(self, idx):

        current_patch = self.coords_all[idx]
        slide = openslide.OpenSlide(self.root + current_patch[0][:-3] + '.svs')
        # print(tuple(current_patch[1]), current_patch[2], tuple([current_patch[3], current_patch[3]]))

        img = slide.read_region(tuple(current_patch[1]), current_patch[2],
                                tuple([current_patch[3], current_patch[3]])).convert('RGB')

	#save_path = "/workdir/rlh353/synthetic_dir/patches/TCGA_brca_WSI_all_CLAM/"
	#img_name = f"{str(current_patch[0][:-3])}_{str(current_patch[1][0])}_{str(current_patch[1][1])}"

        save_path = "/workdir/rlh353/synthetic_dir/patches/TCGA_brca_all_WSI_full_256_patches_Nov7/"

        img_name = f"{str(current_patch[0][:-3])}_{str(current_patch[1][0])}_{str(current_patch[1][1])}"
 #       print(img_name)
    	#saving img
#        img.save(save_path+img_name+".PNG")
#        img.save(save_path+img_name+".RAW")
        img.save(save_path+img_name+".jpg", quality=95)
        return tuple([img, 1])  # current_patch[0][:-3] ])

    def __len__(self):
        return len(self.coords_all)


if __name__ == "__main__":
    path = "/workdir/rlh353/synthetic_dir/TCGA_svs_h5/TCGA_brca_all_WSI_full_256/" 
    tcga_data = TCGA_Dataset(path)
    counter = 0;
    print("length of data: ", len(tcga_data))
    for idx in range(len(tcga_data)): #loops through each index of tcga_data, which represents each batch
        tcga_data[idx] #gets the particular image at index idx and saves it to directory decided in getitem
        counter = counter +1;
        if counter == 500:
            print("Current index: %d",idx)
            counter = 0;

#    print(practice_4.coords_all)
  #  print(len(practice_4.coords_all))
#    print("Practice_4[602][0]")
#    print(practice_4[602][0])
#    practice_4[602][0]
#    print("Length:\n")
#    print(len(practice_4))
#    print("practice_4[330]")
#    practice_4[330]
 #   save_path = "/workdir/rlh353/synthetic_dir/batches/first_hundred_256/"
 #   img_name = "imagebatch_"
        #saving img
 #   img = practice_4[602][0]
 #   img.save(save_path+img_name+".PNG")
#    practice_4[0][0]
#    practice_4[602][0].save("patch602.png")




# a = BLCA_CL_Dataset('E:\\Faltas\\BLCA_H5_HighRes\\')
# = BLCA_CL_Dataset('E:\\Faltas\\BLCA_Cases\\')
# print(len(a))
# print(a[0])

# slides = 'SADULJ-0BKOBY-A2'
"""
#slides = 'TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E'

#slide = openslide.OpenSlide('E:\\BLCA_Diagnostic\\BLCA_Diagnostic_SVS\\' + slides + '.svs')
# print(tuple(current_patch[1]), current_patch[2], tuple([current_patch[3], current_patch[3]]))

#img = slide.read_region(tuple([117264, 53184]), 0, tuple([1024, 1024])).convert('RGB')
#img = img.resize((256, 256))
#img.show()

# c = np.load("E:\\Faltas\\train_list.npy")
# print(c[0])
"""

'''
#a = torchvision.datasets.CIFAR10('./', download = True)

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((300, 300)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

ld = lightly.data.LightlyDataset.from_torch_dataset(a, test_transforms)

collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=300,
    gaussian_blur=0.)


num_ftrs = 512

dataloader_train_simclr = torch.utils.data.DataLoader(
    ld,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)

class MocoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18(pretrained = True)

        resnet.fc = nn.Identity()


        # create a moco based on ResNet
        self.resnet_moco = \
            lightly.models.MoCo(resnet, num_ftrs=512, m=0.99, batch_shuffle=True)

        print(self.resnet_moco)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def forward(self, x):
        self.resnet_moco(x)

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_moco(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()


    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_moco.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

model = MocoModel()

trainer = pl.Trainer(max_epochs=max_epochs, gpus=4,
                     progress_bar_refresh_rate=100)
trainer.fit(
    model,
    dataloader_train_simclr
)

torch.save(model.state_dcit(), '/athena/ihlab/scratch/mbb4001/Bladder/moco_model.pth')
'''

import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils import data
from torchvision import datasets, transforms, utils

from models import create_WALI

cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

class BiGAN_Trainer:
    '''
    Wrapper class for handling the training settings and objects established by CLI
    and created in main.py
    '''
    def __init__(self, args, data, device, image_size):
        self.train_loader = data
        self.device = device
        # path to results dir -- assuming no trailing /
        self.results_dir = args.results_dir
        # training hyperparameters
        self.BATCH_SIZE = args.batch_size
        self.ITER = args.epochs
        self.IMAGE_SIZE = image_size
        self.NUM_CHANNELS = 3
        self.DIM = 128
        self.NLAT = args.latent_dim
        self.LEAK = 0.2
        
        self.C_ITERS = 5       # critic iterations
        self.EG_ITERS = 1      # encoder / generator iterations
        self.LAMBDA = 10       # strength of gradient penalty
        self.LEARNING_RATE = args.lr_adam
        self.BETA1 = 0.5
        self.BETA2 = 0.9

        # flag to enc architecture to include reparam trick or not
        self.determ_enc = args.deterministic_encoder

        # flags for DS, feaD, and feaGE reconstruction loss
        self.DS_recon = args.DS_recon
        self.feaD_recon = args.feaD_recon
        self.feaGE_recon = args.feaGE_recon

        # Patches is the only dataset that doesn't produce labeled data --> this
        # flag will tell us whether or not we need to unpack the elements from the dataloader
        self.LABELED_DATA = (args.dataset != 'patches')
    def train(self):
        wali = create_WALI(img_size=self.IMAGE_SIZE, lru_slope=self.LEAK, num_channels=self.NUM_CHANNELS, dim=self.DIM,
                           nlat=self.NLAT, determ_enc=self.determ_enc, DS_loss=self.DS_recon, feaD_loss=self.feaD_recon,
                           feaGE_loss=self.feaGE_recon).to(self.device)
    
        optimizerEG = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()),
          lr=self.LEARNING_RATE, betas=(self.BETA1, self.BETA2))
        optimizerC = Adam(wali.get_critic_parameters(),
          lr=self.LEARNING_RATE, betas=(self.BETA1, self.BETA2))
    
        noise = torch.randn(self.BATCH_SIZE, self.NLAT, 1, 1, device=self.device)
    
        EG_losses, C_losses = [], []
        curr_iter = C_iter = EG_iter = 0
        C_update, EG_update = True, False
        print('Training starts...')
    
        while curr_iter < self.ITER:
            for batch_idx, x in enumerate(self.train_loader, 1):
                if self.LABELED_DATA:
                    # if we're using one of the labeled datasets, need to separate out the data from the label
                    x = x[0]
                x = x.to(self.device)
    
                if curr_iter == 0:
                    init_x = x
                    curr_iter += 1
    
                z = torch.randn(x.size(0), self.NLAT, 1, 1).to(self.device)
                C_loss, EG_loss = wali(x, z, lamb=self.LAMBDA)
    
                if C_update:
                    optimizerC.zero_grad()
                    C_loss.backward()
                    C_losses.append(C_loss.item())
                    optimizerC.step()
                    C_iter += 1
    
                    if C_iter == self.C_ITERS:
                        C_iter = 0
                        C_update, EG_update = False, True
                    continue
    
                if EG_update:
                    optimizerEG.zero_grad()
                    EG_loss.backward()
                    EG_losses.append(EG_loss.item())
                    optimizerEG.step()
                    EG_iter += 1
    
                    if EG_iter == self.EG_ITERS:
                        EG_iter = 0
                        C_update, EG_update = True, False
                        curr_iter += 1
                    else:
                        continue
    
                # print training statistics
                if curr_iter % 100 == 0:
                    print('[%d/%d]\tW-distance: %.4f\tC-loss: %.4f'
                      % (curr_iter, self.ITER, EG_loss.item(), C_loss.item()))
    
                    # plot reconstructed images and samples
                    wali.eval()
                    real_x, rect_x = init_x[:self.BATCH_SIZE//2], wali.reconstruct(init_x[:self.BATCH_SIZE//2]).detach_()
                    rect_imgs = torch.cat((real_x.unsqueeze(1), rect_x.unsqueeze(1)), dim=1)
                    rect_imgs = rect_imgs.view(self.BATCH_SIZE, self.NUM_CHANNELS, self.IMAGE_SIZE, self.IMAGE_SIZE).cpu()
                    genr_imgs = wali.generate(noise).detach_().cpu()
                    utils.save_image(rect_imgs * 0.5 + 0.5, '%s/rect%d.png' % (self.results_dir, curr_iter))
                    utils.save_image(genr_imgs * 0.5 + 0.5, '%s/genr%d.png' % (self.results_dir, curr_iter))
                    wali.train()
    
                # save model
                if curr_iter % (self.ITER // 10) == 0:
                    torch.save(wali.state_dict(), '%s/models/%d.ckpt' % (self.results_dir, curr_iter))
    
        # plot training loss curve
        plt.figure(figsize=(10, 5))
        plt.title('Training loss curve')
        plt.plot(EG_losses, label='Encoder + Generator')
        plt.plot(C_losses, label='Critic')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('%s/loss_curve.png' % self.results_dir)

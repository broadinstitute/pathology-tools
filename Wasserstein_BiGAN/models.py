'''
File containing the different model classes for the W-BiGAN
'''
import torch
import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
from util import DeterministicConditional, GaussianConditional, JointCritic, WALI

def create_encoder(img_size=32, NUM_CHANNELS=3, DIM=128, NLAT=256, determ=True):
    mapping = None
    if img_size == 32:
        mapping = nn.Sequential(
          Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
          Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
          Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
          Conv2d(DIM * 4, DIM * 4, 4, 1, 0, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
          Conv2d(DIM * 4, NLAT, 1, 1, 0))
    elif img_size == 64:
        mapping = nn.Sequential(
          Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
          Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
          Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
          Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
          Conv2d(DIM * 8, DIM * 8, 4, 1, 0, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
          Conv2d(DIM * 8, NLAT, 1, 1))
    elif img_size == 128:
        mapping = nn.Sequential(
          Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
          Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
          Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
          Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
          Conv2d(DIM * 8, DIM * 16, 4, 2, 1, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
          Conv2d(DIM * 16, DIM * 16, 4, 1, 0, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
          Conv2d(DIM * 16, NLAT, 1, 1))
    else:
        raise ValueError('Must specify image size of 32, 64, or 128')
    # Setting E to be a deterministic conditional doesn't employ the reparameterization trick
    # --> This regime has the network learn the latent distribution implicitly, NOT through the parameters
    if determ:
        return DeterministicConditional(mapping)
    else:
        return GaussianConditional(mapping)

def create_generator(img_size=32, NUM_CHANNELS=3, DIM=128, NLAT=256):
    mapping = None
    if img_size == 32:
        mapping = nn.Sequential(
          ConvTranspose2d(NLAT, DIM * 4, 4, 1, 0, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
          ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
          ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
          ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), Tanh())
    elif img_size == 64:
        mapping = nn.Sequential(
          ConvTranspose2d(NLAT, DIM * 8, 4, 1, 0, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
          ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
          ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
          ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
          ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), Tanh())
    elif img_size == 128:
        mapping = nn.Sequential(
          ConvTranspose2d(NLAT, DIM * 16, 4, 1, 0, bias=False), BatchNorm2d(DIM * 16), ReLU(inplace=True),
          ConvTranspose2d(DIM * 16, DIM * 8, 4, 2, 1, bias=False), BatchNorm2d(DIM * 8), ReLU(inplace=True),
          ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
          ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
          ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
          ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), Tanh())
    else:
        raise ValueError('Must specify image size of 32, 64, or 128')
    return DeterministicConditional(mapping)

def create_critic(img_size=32, LEAK=0.2, NUM_CHANNELS=3, DIM=128, NLAT=256):
    x_mapping, z_mapping, joint_mapping = None, None, None
    if img_size == 32:
        x_mapping = nn.Sequential(
          Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
          Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
          Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
          Conv2d(DIM * 4, DIM * 4, 4, 1, 0), LeakyReLU(LEAK))
        z_mapping = nn.Sequential(
          Conv2d(NLAT, 512, 1, 1, 0), LeakyReLU(LEAK),
          Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK))
        joint_mapping = nn.Sequential(
          Conv2d(DIM * 4 + 512, 1024, 1, 1, 0), LeakyReLU(LEAK),
          Conv2d(1024, 1024, 1, 1, 0), LeakyReLU(LEAK),
          Conv2d(1024, 1, 1, 1, 0))
    elif img_size == 64:
        # critic network taken from the celebA example
        x_mapping = nn.Sequential(
          Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
          Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
          Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
          Conv2d(DIM * 4, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
          Conv2d(DIM * 8, DIM * 8, 4, 1, 0), LeakyReLU(LEAK))
        z_mapping = nn.Sequential(
          Conv2d(NLAT, 512, 1, 1, 0), LeakyReLU(LEAK),
          Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK))
        joint_mapping = nn.Sequential(
          Conv2d(DIM * 8 + 512, 2048, 1, 1, 0), LeakyReLU(LEAK),
          Conv2d(2048, 2048, 1, 1, 0), LeakyReLU(LEAK),
          Conv2d(2048, 1, 1, 1, 0))
    elif img_size == 128:
        x_mapping = nn.Sequential(
          Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
          Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
          Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
          Conv2d(DIM * 4, DIM * 8, 4, 2, 1), LeakyReLU(LEAK),
          Conv2d(DIM * 8, DIM * 16, 4, 2, 1), LeakyReLU(LEAK),
          Conv2d(DIM * 16, DIM * 16, 4, 1, 0, bias=False), LeakyReLU(LEAK))
        z_mapping = nn.Sequential(
          Conv2d(NLAT, 512, 1, 1, 0), LeakyReLU(LEAK),
          Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK))
        # not sure what the output dimension should be for the first conv layer here
        # --> for the 64x64 network they use 2048 as the internal dimension, so might
        # --> want to go even bigger for 128x128
        joint_mapping = nn.Sequential(
          Conv2d(DIM * 16 + 512, 1024, 1, 1, 0), LeakyReLU(LEAK),
          Conv2d(1024, 1024, 1, 1, 0), LeakyReLU(LEAK),
          Conv2d(1024, 1, 1, 1, 0))
    else:
        raise ValueError('Must specify image size of 32 or 128')
    return JointCritic(x_mapping, z_mapping, joint_mapping)

def create_WALI(img_size=32, lru_slope=0.2, num_channels=3, dim=128, nlat=256, determ_enc=True, DS_loss=False,
                feaD_loss=False, feaGE_loss=False):
    '''
    Instantiates the WALI network with the versions of the subnetworks
    that match the image size we'll be getting from the dataset
    - determ_enc: flag for using DeterministicConditional or GaussianConditional for the encoder
    '''
    E = create_encoder(img_size, num_channels, dim, nlat, determ=determ_enc)
    G = create_generator(img_size, num_channels, dim, nlat)
    C = create_critic(img_size, lru_slope, num_channels, dim, nlat)
    wali = WALI(E, G, C, DS_loss, feaD_loss, feaGE_loss)
    return wali

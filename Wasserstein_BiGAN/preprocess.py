'''
Script to handle dataset/dataloader creation
'''
import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Subset
from patch_dataset import BLCA_CL_Dataset

def get_svhn(args, size=128, data_dir='./data/svhn/', dataset_size=None, num_workers=None):
    """Returning svhn dataloder."""
    transform = transforms.Compose([transforms.Resize(size),                
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = datasets.SVHN(root=data_dir, download=True, transform=transform)
    if dataset_size:
        data = Subset(data, list(range(dataset_size)))
    if num_workers:
        dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    else:
        dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    return dataloader

def get_cifar(args, size=128, data_dir='./data/cifar/', dataset_size=None, num_workers=None):
    """Returning cifar dataloder."""
    transform = transforms.Compose([transforms.Resize(size),                
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = datasets.CIFAR10(root=data_dir, download=True, transform=transform)
    if dataset_size:
        data = Subset(data, list(range(dataset_size)))
    if num_workers:
        dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    else:
        dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    return dataloader

def get_patches(args, size=128, dataset_size=None, num_workers=None):
    # not sure if the same normalization will make sense with the patches, but trying here just to have this
    # involve *only* a change in the images themselves
    transform = transforms.Compose([transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Patch dataset object takes optional transform parameter -- if provided, it's used in the __getitem__ method
    # if it's not provided, the __getitem__ method just uses the ToTensor transformation
    data = BLCA_CL_Dataset('/workdir/crohlice/software/CLAM/TCGA_svs_h5_128/', mode='Train', transform=transform)
    if dataset_size:
        data = Subset(data, list(range(dataset_size)))
    if num_workers:
        dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    else:
        dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    return dataloader

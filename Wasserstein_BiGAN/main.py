'''
Script to process commandline input arguments regarding
training setup, and pass them to the training script
'''
import argparse
import torch
from preprocess import get_svhn, get_cifar, get_patches
from train import BiGAN_Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['patches', 'svhn', 'cifar'], default='svhn',
                        help='Dataset - options: patches, cifar, svhn')
    parser.add_argument('--image_size', type=int, choices=[32, 64, 128], default=128)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of epochs')
    parser.add_argument('--lr_adam', type=float, default=1e-4, help='Base learning rate for Adam')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of latent variable z')
    parser.add_argument('--results_dir', default='./scratch_results')
    parser.add_argument('--dataset_workers', type=int, default=0, help='num_workers for torch dataloader')
    parser.add_argument('--dataset_size', type=int, default=None, help='optional limiting dataset size')
    parser.add_argument('--gpu_id', choices=['0', '1', '2', '3'], default='0', help='GPU where training will be run')
    parser.add_argument('--deterministic_encoder', type=bool, default=True,
                        help='switch for using a deterministic- or gaussian-output encoder')
    parser.add_argument('--patch_dataset_source_path', type=str,
                        default='/workdir/crohlice/software/CLAM/TCGA_svs_h5_128/',
                        help='path to grab the CLAM patch images needed to create the patch dataloader')
    parser.add_argument('--DS_recon', type=bool, default=False,
                        help='flag for adding DS reconstruction loss to G/E optimization')
    parser.add_argument('--feaD_recon', type=bool, default=False,
                        help='flag for adding feaD reconstruction loss to G/E optimization')
    parser.add_argument('--feaGE_recon', type=bool, default=False,
                        help='flag for adding GE reconstruction loss to G/E optimization')

    args = parser.parse_args()
    device = torch.device('cuda:%s' % args.gpu_id if torch.cuda.is_available() else 'cpu')
    num_workers = None
    if args.dataset_workers > 0:
        num_workers = args.dataset_workers
    # creating dataloader based on input
    # --> also storing image size variable based on which dataset was chose
    data = None
    if args.dataset == 'svhn':
        data = get_svhn(args, size=args.image_size, dataset_size=args.dataset_size, num_workers=num_workers)
    elif args.dataset == 'cifar':
        data = get_cifar(args, size=args.image_size, dataset_size=args.dataset_size, num_workers=num_workers)
    elif args.dataset == 'patches':
        data = get_patches(args, size=args.image_size, dataset_size=args.dataset_size, num_workers=num_workers,
                           source_path=args.patch_dataset_source_path)
    else:
        raise ValueError('Dataset not set properly')

    bigan = BiGAN_Trainer(args, data, device, args.image_size)
    bigan.train()

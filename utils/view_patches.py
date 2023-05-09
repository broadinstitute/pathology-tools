"""
util script to visualize samples from .h5 files containing patch images
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py


def view_patches(h5_file, num_patches, output_dir, shuffle=False):
    dataset = h5py.File(h5_file)
    imgs = dataset['images']
    inds = np.random.choice(len(imgs), num_patches) if shuffle else range(num_patches)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in inds:
        plt.imsave(f'{output_dir}/patch_{i}.png', imgs[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helper script for quick visualization of patches from h5 files')
    parser.add_argument('--input_h5', type=str, help='input .h5 file containing patches to be visualized')
    parser.add_argument('--num_patches', type=int, help='number of patches to be viewed from dataset')
    parser.add_argument('--output_dir', type=str, help='output image directory')
    parser.add_argument('--shuffle', action='store_true', help='bool flag for drawing random patches from the file')
    args = parser.parse_args()
    view_patches(args.input_h5, args.num_patches, args.output_dir, shuffle=args.shuffle)

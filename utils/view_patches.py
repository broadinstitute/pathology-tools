"""
util script to visualize samples from .h5 files containing patch images
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import time

from PIL import Image

def view_patches_matplotlib(h5_file, num_patches, output_dir, shuffle=False):
    dataset = h5py.File(h5_file)
    imgs = dataset['images']
    if num_patches == 'all':
        print(f'Generating images for all {len(imgs)} samples')
        inds = range(len(imgs))
    else:
        inds = np.random.choice(len(imgs), int(num_patches)) if shuffle else range(int(num_patches))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in inds:
        plt.imsave(f'{output_dir}/MPL_patch_{i}.png', imgs[i])


def view_patches_PIL(h5_file, num_patches, output_dir, shuffle=False):
    dataset = h5py.File(h5_file)
    imgs = dataset['images']
    if num_patches == 'all':
        print(f'Generating images for all {len(imgs)} samples')
        inds = range(len(imgs))
    else:
        inds = np.random.choice(len(imgs), int(num_patches)) if shuffle else range(int(num_patches))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in inds:
        img = Image.fromarray(imgs[i], 'RGB')
        img.save(f'{output_dir}/PIL_patch_{i}.png', 'PNG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helper script for quick visualization of patches from h5 files')
    parser.add_argument('--input_h5', type=str, help='input .h5 file containing patches to be visualized')
    parser.add_argument('--num_patches', help='number of patches to be viewed from dataset (expected integer, or'
                                              '\'all\')')
    parser.add_argument('--output_dir', type=str, help='output image directory')
    parser.add_argument('--shuffle', action='store_true', help='bool flag for drawing random patches from the file')
    args = parser.parse_args()

    start_time = time.time()
    view_patches_matplotlib(args.input_h5, args.num_patches, args.output_dir, shuffle=args.shuffle)
    mpl_time = time.time()
    view_patches_PIL(args.input_h5, args.num_patches, args.output_dir, shuffle=args.shuffle)
    pil_time = time.time()
    # debug -- time comparison for image conversion
    print(f'---- Time to generate {args.num_patches} images ----\nmatplotlib: {mpl_time - start_time} seconds'
          f'\nPIL: {pil_time - mpl_time} seconds')

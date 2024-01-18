import os
import argparse
import pandas as pd
import numpy as np
import scipy.io as sio

from tqdm import tqdm


def extract_purity(mat_dirs, output_path):
    """
    Calculates purity from hovernet output .mat files given in mat_dirs
    Writes results to a .csv file at output_path
    """
    mat_files = [os.path.join(p, fname) for p in mat_dirs for fname in os.listdir(p)]
    cancer_purity = []
    for mat_file in tqdm(mat_files):
        seg = sio.loadmat(mat_file)
        unique, counts = np.unique(seg['inst_type'], return_counts=True)
        # dict keyed on hovernet labels - 1: cc, 2: imm, 3: con, 4: nec, 5: ep
        counts_dict = {}
        for cell_type in range(1, 6):
            idx = np.argwhere(unique == cell_type)
            counts_dict[cell_type] = counts[idx[0, 0]] if idx.shape[0] == idx.shape[1] == 1 else 0
        pur = round(counts_dict[1] / seg['inst_type'].shape[0], 4) if seg['inst_type'].shape[0] > 0 else 0
        cancer_purity.append(pd.DataFrame(np.asarray([mat_file[:-4]+'.png',
                                                      counts_dict[1], pur, counts_dict[2], counts_dict[3],
                                                      counts_dict[4], counts_dict[5]], dtype=object).reshape((1, -1))))
    cp_df = pd.concat(cancer_purity)
    cp_df.to_csv(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparse for garbage function')
    parser.add_argument('--mat_paths', nargs='+', help='list of paths to .mat file directories')
    parser.add_argument('--output_path', help='output path for cancer purity .csv file')
    args = parser.parse_args()

    extract_purity(args.mat_paths, args.output_path)

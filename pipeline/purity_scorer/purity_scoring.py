import pandas as pd
import numpy as np
import os
import itertools
import shutil
import scipy.io as sio
from tqdm import tqdm
#./BLCA_SVS_Cases_Seg/mat/

seg = os.listdir('/local/storage/TCGA_data/hovernet_output/pathologyGAN/all_brca_patchlevel0_epoch300/mat/')
#seg = os.listdir('./BLCA_SVS_Cases_Seg/mat/')
#seg2 = os.listdir('./BLCA_SVS_Cases_Seg2/mat/')
#seg3 = os.listdir('./BLCA_SVS_Cases_Seg3/mat/')
#seg4 =  os.listdir('./BLCA_SVS_Cases_Seg4/mat/')

#allf1 = os.listdir('./BLCA_SVS_Cases_Tiles_Pass/')
#allf2 = os.listdir('./BLCA_SVS_Cases_Tiles_Pass2')
#allf3 =  os.listdir('./BLCA_SVS_Cases_Tiles_Pass3')

#f = list(itertools.chain(allf1,allf2,allf3))

#print(len(list(set(f))))

#print(f[:1])

#f_new = [x[:-4] for x in f]

import itertools
ab = list(itertools.chain(seg))
#ab = list(itertools.chain(seg, seg2,seg3,seg4))

print(len(ab), len(list(set(ab))))

fin_ab = list(set(ab))


cancer_purity = []

folds = ['/local/storage/TCGA_data/hovernet_output/pathologyGAN/all_brca_patchlevel0_epoch300/mat/']
#folds = ['/athena/ihlab/scratch/mbb4001/SingleCellSegment/BLCA_SVS_Cases_Seg/mat/', '/athena/ihlab/scratch/mbb4001/SingleCellSegment/BLCA_SVS_Cases_Seg2/mat/', '/athena/ihlab/scratch/mbb4001/SingleCellSegment/BLCA_SVS_Cases_Seg3/mat/', '/athena/ihlab/scratch/mbb4001/SingleCellSegment/BLCA_SVS_Cases_Seg4/mat/']
for i in tqdm(fin_ab):#[:5]:
    #print(i + '.mat')
    test1 = os.path.exists(f'/local/storage/TCGA_data/hovernet_output/pathologyGAN/all_brca_patchlevel0_epoch300/mat/{i}')
    #test1 = os.path.exists(f'/athena/ihlab/scratch/mbb4001/SingleCellSegment/BLCA_SVS_Cases_Seg/mat/{i}')    
    #test2 = os.path.exists(f'/athena/ihlab/scratch/mbb4001/SingleCellSegment/BLCA_SVS_Cases_Seg2/mat/{i}')
    #test3 = os.path.exists(f'/athena/ihlab/scratch/mbb4001/SingleCellSegment/BLCA_SVS_Cases_Seg3/mat/{i}')
    #test4 = os.path.exists(f'/athena/ihlab/scratch/mbb4001/SingleCellSegment/BLCA_SVS_Cases_Seg4/mat/{i}')

    checks = [test1]
    #checks = [test1,test2,test3,test4]
    first = [i for i, x in enumerate(checks) if x] #next(filter(lambda check: check == True, checks), None)
    #print(checks,first)
    
    seg = sio.loadmat(folds[first[0]]+i)

    #print(seg.keys())
    #print(seg['inst_type'], i)


    unique, counts = np.unique(seg['inst_type'], return_counts=True)
    canc_idx = np.argwhere(unique == 1)
    
    if canc_idx.shape[0] > 0:
        #print(canc_idx.shape)
        #print(np.asarray((unique, counts)).T)
        #print(counts[canc_idx[0,0]])
        assert canc_idx.shape[0] == 1 and  canc_idx.shape[1] == 1
        cc = counts[canc_idx[0,0]]
        #print(seg['inst_type'].shape[0])
    else:
        cc = 0
    if seg['inst_type'].shape[0] > 0:
        pur = round(cc/seg['inst_type'].shape[0],4)
    else:
        pur = 0


    imm_idx = np.argwhere(unique == 2)

    if imm_idx.shape[0] > 0:
        assert imm_idx.shape[0] == 1 and  imm_idx.shape[1] == 1
        ic = counts[imm_idx[0,0]]
        
    else:
        ic = 0
    
    con_idx = np.argwhere(unique == 3)

    if con_idx.shape[0] > 0:
        assert con_idx.shape[0] == 1 and  con_idx.shape[1] == 1
        conc = counts[con_idx[0,0]]
        
    else:
        conc = 0
    
    nec_idx = np.argwhere(unique == 4)

    if nec_idx.shape[0] > 0:
        assert nec_idx.shape[0] == 1 and  nec_idx.shape[1] == 1
        nc = counts[nec_idx[0,0]]

    else:
        nc = 0

    ep_idx = np.argwhere(unique == 5)

    if ep_idx.shape[0] > 0:
        assert ep_idx.shape[0] == 1 and  ep_idx.shape[1] == 1
        ec = counts[ep_idx[0,0]]

    else:
        ec = 0



    cancer_purity.append(pd.DataFrame(np.asarray([i[:-4]+'.png',cc, pur,ic,conc,nc,ec ],dtype = object).reshape((1,-1))))
 
cp_df = pd.concat(cancer_purity)
cp_df.to_csv('/local/storage/TCGA_data/purity_scores/pathologyGAN/all_brca_patchlevel0_epoch300/purity_score.csv')
#cp_df.to_csv('./BLCA_Purity.csv')

#ab_new = [x[:-4] for x in ab]

#print(list(set(f_new).difference(ab_new)))
#assert len(list(set(ab))) == len(ab)

#missing_all = list(set(f_new).difference(ab_new))
#print(missing)
#for missing in missing_all:
#    print() 
#    test1 = os.path.exists(f'/athena/ihlab/scratch/mbb4001/SingleCellSegment/BLCA_SVS_Cases_Tiles_Pass/{missing}.jpg') 
    #test2 =  os.path.exists(f'/athena/ihlab/scratch/mbb4001/SingleCellSegment/BLCA_SVS_Cases_Tiles_Pass2/{missing}.jpg')
    #test3 =  os.path.exists(f'/athena/ihlab/scratch/mbb4001/SingleCellSegment/BLCA_SVS_Cases_Tiles_Pass3/{missing}.jpg')
    #print(test1,test2,test3)
    #assert test1+test2+test3 == 1

#    shutil.copy(f'/athena/ihlab/scratch/mbb4001/SingleCellSegment/BLCA_SVS_Cases_Tiles_Pass/{missing}.jpg', f'/athena/ihlab/scratch/mbb4001/SingleCellSegment/BLCA_SVS_Cases_Tiles_Pass4/')
   

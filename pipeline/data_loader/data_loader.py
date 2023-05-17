
import configparser
from tqdm import tqdm
import pandas as pd
import os
import math
import numpy as np
import PIL
import time

import math
from random import gauss



# Function to read and parse config file
def read_config(file_path):
    bins=['10','20','30','40','50','60','70','80','90','100']
    t_config = {}
    config = configparser.ConfigParser()
    config.read(file_path)
    for section in config.sections():
        for key in config[section]:
            t_config[key] = config[section][key]
            
    # Change boolean type
    if t_config['replace']=='False':
        t_config['replace']=False
    else:
        t_config['replace']=True
    
    if t_config['use_residuals']=='False':
        t_config['use_residuals']=False
    else:
        t_config['use_residuals']=True
        
    # Make purities into list and check list 
    p_bins = []
    for t_bin in bins:
        t_config[t_bin] = t_config[t_bin].split(",")
        
        # Check if zero weights are given
        if '0' in t_config[t_bin]:
            raise ValueError('Weight cannot be zero!')
        
        p_bins.append(t_config[t_bin])
    
    # Check list size
    it = iter(p_bins)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError('Bin list must have same length!')
                
    return t_config
 

 ######
# Function to bin purities by decile
def bin_purity(row):
    if 0.0 <= row['Purity'] <= 0.1:
        return 10
    elif 0.1 < row['Purity'] <= 0.2:
        return 20
    elif 0.2 < row['Purity'] <= 0.3:
        return 30
    elif 0.3 < row['Purity'] <= 0.4:
        return 40
    elif 0.4 < row['Purity'] <= 0.5:
        return 50
    elif 0.5 < row['Purity'] <= 0.6:
        return 60
    elif 0.6 < row['Purity'] <= 0.7:
        return 70
    elif 0.7 < row['Purity'] <= 0.8:
        return 80
    elif 0.8 < row['Purity'] <= 0.9:
        return 90
    return 100


# Function to assemble slide
# Params: 
# patch_purities -> main table with patch level purities
# cancer_types -> list of cancer cohorts
# Returns: Dictionary of cancer_type -> dict of slides -> tiles
def assemble_slides(patch_purities, cancer_types):
    slides={}
    # slide counter per cohort 
    i=1
    for cohort in cancer_types:
        # number of patches per cohort
        total_patches = len(patch_purities[patch_purities['CancerType']==cohort])
        # print(total_patches)
        # tiles per slide threshold check:
        if (total_patches >= int(config['tiles_per_slide'])):
            print(cohort,":", len(patch_purities[patch_purities['CancerType']==cohort]), "patches")
            slides[cohort]={}
            while total_patches>=int(config['tiles_per_slide']):
                # sample and remove
                # print(len(patch_purities['CancerType']==cohort))
                slide_sample = patch_purities[patch_purities['CancerType']==cohort].sample(
                    n=int(config['tiles_per_slide']),
                    replace=config['replace'],
                    weights=patch_purities['Weight'],
                    random_state=37
                )
                patch_purities=patch_purities.drop(slide_sample.index)
                total_patches = len(patch_purities[patch_purities['CancerType']==cohort])
                slides[cohort]['SYN-'+cohort+'-'+str(i)]=slide_sample
                print('SYN-'+cohort+'-'+str(i),"slide assembled with", len(slide_sample), "patches")
                i+=1

            # check tiles threshold to discard or add remaining slides based on flag
            if config['use_residuals'] and total_patches >= int(config['tiles_thres']):
                slide_sample = patch_purities[patch_purities['CancerType']==cohort]
                patch_purities=patch_purities.drop(slide_sample.index)
                total_patches = len(patch_purities[patch_purities['CancerType']==cohort])
                assert(total_patches==0)
                slides[cohort]['SYN-'+cohort+'-'+str(i)]=slide_sample
                print('SYN-'+cohort+'-'+str(i),"slide assembled with", len(slide_sample), "residual patches")
                # reset index
                i=1
            else:
                print(cohort,": discarded", total_patches, "patches. (Thres):",config['tiles_thres'])
                patch_purities=patch_purities.drop(patch_purities[patch_purities['CancerType']==cohort].index)
                total_patches = len(patch_purities[patch_purities['CancerType']==cohort])
                assert(total_patches==0)
                # reset index
                i=1


        else:
            print(cohort,":", len(patch_purities[patch_purities['CancerType']==cohort]), "patches")
            # check residual flag and threshold to add or discard remaining slides
            if config['use_residuals'] and total_patches >= int(config['tiles_thres']):
                slides[cohort]={}
                slide_sample = patch_purities[patch_purities['CancerType']==cohort]
                patch_purities=patch_purities.drop(slide_sample.index)
                total_patches = len(patch_purities[patch_purities['CancerType']==cohort])
                assert(total_patches==0)
                slides[cohort]['SYN-'+cohort+'-'+str(i)]=slide_sample
                print('SYN-'+cohort+'-'+str(i),"slide assembled with", len(slide_sample), "residual patches")

            else:
                print(cohort,": discarded", total_patches, "patches. (Thres):",config['tiles_thres'])
                patch_purities=patch_purities.drop(patch_purities[patch_purities['CancerType']==cohort].index)
                total_patches = len(patch_purities[patch_purities['CancerType']==cohort])
                assert(total_patches==0)
                
    return slides
    

if __name__ == "__main__":
    # Read config file
    config = read_config("./config_list.ini")
    
    ###
    # read patch purity file
    patch_purities = pd.read_csv(config['patch_purity_file'])  
    # Matt's file: index,image name,cc,pur,ic,conc,nc,ec
    patch_purities = patch_purities.loc[:,["0","2",]]
    patch_purities = patch_purities.rename(columns={'0':'img','2':'Purity'})
    patch_purities
    # # sort by purity
    # patch_purities = patch_purities.sort_values(by=['Purity'])

    # Bin patches
    patch_purities['Bin']=patch_purities.apply(lambda row:bin_purity(row),axis=1)

    ###
    patch_purities['Weight'] = patch_purities['Bin'].map({
        10: config['10'],
        20: config['20'],
        30: config['30'],
        40: config['40'],
        50: config['50'],
        60: config['60'],
        70: config['70'],
        80: config['80'],
        90: config['90'],
        100: config['100']
    })

    # testing only one cohort for now
    patch_purities['CancerType'] = 'Breast'
    cancer_types=['Breast']

    # Loop through all purities in list and generate slides
    slides_list = {}
    for i in range (0,len(config['10'])):
        # assign weights
        patch_purities['Weight'] = patch_purities['Bin'].map({
        10: config['10'][i],
        20: config['20'][i],
        30: config['30'][i],
        40: config['40'][i],
        50: config['50'][i],
        60: config['60'][i],
        70: config['70'][i],
        80: config['80'][i],
        90: config['90'][i],
        100: config['100'][i]
        })

        
        patch_purities['Weight']=patch_purities['Weight'].astype(str).astype(int)

        slides = assemble_slides(patch_purities, cancer_types)
        slides_list['config-'+str(i)] = slides

    for key, value in slides_list.items() :
        print(key)
        for key1, value1 in value.items():
            print ('\t',key1)
            for key2, value2 in value1.items():
                print ('\t\t',key2)


    # Move patches as slides for ws-purity 
    # new code for copying

    # Patch image directory format:
    # Cancer_type_folder/SLIDE_NAME_FOLDER/TILE_NAME.JPG

    # variables
    # original path where the patches are stored
    patches_path = config['patches_dir']
    # hard coded: 
    # patches_path = '/local/storage/TCGA_data/pathologyGAN_synthetic_output_all_brca/generated_images/'
    # locally
    # patches_path = 'hovernet/pgan_synthetic'

    # path to create synthetic slides for ws-purity
    slide_path = config['slide_path']

    for config in slides_list:
        os.mkdir(f'{slide_path}/{config}')
        for cancer_type in slides_list[config]:
            os.mkdir(f'{slide_path}/{config}/{cancer_type}')
            for slide in slides_list[config][cancer_type]:
                os.mkdir(f'{slide_path}/{config}/{cancer_type}/{slide}')
                for idx, row in slides_list[config][cancer_type][slide].iterrows():
                    tile = row['img']
                    # if it is jpg
                    # tile = row['img'].split('.')[0]+'.jpg'
                    # print(tile)
                    os.popen(f'cp {patches_path}/{tile} {slide_path}/{config}/{cancer_type}/{slide}/{tile}')
                    

    # CSV file prep for ws-purity 
    ## (TileClusterSubset_120.csv,TissueType.csv, TP_Purity_list.csv)

    # #### DBSCAN clusters (TileClusterSubset_120)
    # Column names:
    #     ,	Slide,	Tile,	Cluster,	Subset


    # #### TissueType.csv
    # index, slide_name, cancer_type
    # Column names: 
    #     , 0, 1

    # #### Data split csv (TP_Purity_list.csv)
    # index,slide_name, caseID (slide_name), tumor purity, dataset (train/test/validation)
    # Column names:
    #     , Slide_SVS_Name, CaseID,	TP_VALUE,	Dataset


    for config in slides_list:
        dbscan_dict={}
        tissue_type_dict={}
        data_split_dict={}
        i=0
        j=0
        for cancer_type in slides_list[config]:
            for slide in slides_list[config][cancer_type]:
                tissue_type_dict[i]={'0':slide,'1':cancer_type}
                tp_val = 0
                for idx, row in slides_list[config][cancer_type][slide].iterrows():
                    tile = row['img']
                    # if it is jpg
                    # tile = row['img'].split('.')[0]+'.jpg'
                    tp_val+=row['Purity']
                    dbscan_dict[j]={'Slide':slide,'Tile':tile,'Cluster':0,'Subset':0}
                    j+=1
                # purity avg of assembled slide (should be equal to the theoretical weighted avg)
                tp_val/=len(slides_list[config][cancer_type][slide])
                data_split_dict[i]={'Slide_SVS':slide,'CaseID':slide,'TP_VALUE':tp_val,'Dataset':'test'}

                i+=1


        dbscan_file = pd.DataFrame.from_dict(dbscan_dict,orient='index')
        tissue_type_file = pd.DataFrame.from_dict(tissue_type_dict,orient='index')
        data_split_file = pd.DataFrame.from_dict(data_split_dict,orient='index')

        dbscan_file.to_csv(f'{slide_path}/{config}/TileClusterSubset_120_syn.csv',index=True)
        tissue_type_file.to_csv(f'{slide_path}/{config}/TissueType_syn.csv',index=True)
        data_split_file.to_csv(f'{slide_path}/{config}/TP_Purity_list_syn.csv',index=True)
                    







    



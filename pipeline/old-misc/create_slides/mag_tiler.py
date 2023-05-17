import openslide
import numpy as np
from openslide.deepzoom import *
import cv2
import PIL
import time
from tqdm import tqdm
import pandas as pd
import os
import math
import haar

svs_path = '/local/storage/TCGA_data/merge_folder' # path to svs images

tiles_inc = {} # included tiles per mag
tiles_exc = {} # excluded tiles per mag

# mag: Desired magnification level
def mag_tile(mag):
    tile_path = f'tiles/{mag}x' # path to store tiles

    try:
        os.mkdir(f'tiles/{mag}x')
    except:
        pass    

    tiles_inc[f'{mag}x'] = {} # included tiles for current mag
    tiles_exc[f'{mag}x'] = {} # excluded tiles for current mag

    # loop through svs images
    for file in tqdm(os.listdir(svs_path)):

        # confirm svs file type
        if 'svs' not in file:
            continue

        slidename = file[:-4]

        # skip completed images
        try:
            os.mkdir(f'{tile_path}/{slidename}/') # path to tiles for current image
        except:
            continue

#        print(slidename, 'Initiated!')

        try:
            slide = openslide.OpenSlide(f'{svs_path}/{slidename}.svs')

            # tile size with 0 overlap
            tile_size = int(1024/(40/mag))  # downsampling
            zoom = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)

            highest_level = zoom.level_count-1

            # get level corresponding to desired magnification, or the closest available
            obj_power = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            offset = int(math.log2(obj_power/mag))  # calculate offset by factor of 2
            level = highest_level - offset

            levels = zoom.level_count
            tiles = zoom.level_tiles[level]

            # loop through each tile
            for X in range(tiles[0]):
                for Y in range(tiles[1]):
                    tile = zoom.get_tile(level, (X, Y))

                    # fill smaller tiles with white bg for size consistency
                    rgb = np.array(tile, dtype=float)
                    bgr = rgb[..., ::-1]
                    dim = bgr.shape

                    if dim[0] < tile_size:
                        white_row = np.zeros((1, dim[1], 3)) + 255
                        white_rows = np.broadcast_to(white_row, (tile_size - dim[0], dim[1], 3))
                        bgr = np.concatenate((bgr, white_rows), axis=0)
                    dim = bgr.shape

                    if dim[1] < tile_size:
                        white_col = np.zeros((dim[0], 1, 3)) + 255
                        white_cols = np.broadcast_to(white_col, (dim[0], tile_size-dim[1], 3))
                        bgr = np.concatenate((bgr, white_cols), axis=1)
                    bgr = bgr.astype('uint8')

                    # change color scheme
                    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

                    image_area = np.size(bgr[:, :, 0])

                    # thresholds for hsv color scheme
                    # lower thres is towards bg
                    # upper thres is towards tissue
                    lower = (122, 30, 0)
                    upper = (179, 255, 255)

                    # check tissue area in tile (40% threshold) otw discard
                    tissue_area = np.count_nonzero(
                        cv2.inRange(hsv, lower, upper))
                    
                    if tissue_area > (0.40*image_area):
                        final = cv2.resize(bgr, (tile_size, tile_size))                        
                        cv2.imwrite(f'{tile_path}/{slidename}/{slidename}_{X}_{Y}.jpg', final)
                        tiles_inc[f'{mag}x'][f'{slidename}_{X}_{Y}'] = (tissue_area, 0.40*image_area)
                            
                    else:
                        tiles_exc[f'{mag}x'][f'{slidename}_{X}_{Y}'] = (tissue_area, 0.40*image_area)

        except Exception as e:
#            print(e)
#            print(slidename, 'Unsuccessful!')
            continue

#        print(slidename, 'Completed!')

mag_tile(5)
mag_tile(10)
mag_tile(20)
mag_tile(40)

'''
for tile in tiles_exc['40x']:
    
    if tile not in tiles_exc['10x']:
        print('40/20x exc, 10/5x inc:')
        print('40x exc:', tile, tiles_exc['40x'][tile])
        print('20x exc:', tile, tiles_exc['20x'][tile])
        print('10x inc:', tile, tiles_inc['10x'][tile])
        print('5x inc:', tile, tiles_inc['5x'][tile])
        print('')
        
    elif tile not in tiles_exc['5x']:
        print('40/20/10x exc, 5x inc:')
        print('40x exc:', tile, tiles_exc['40x'][tile])
        print('20x exc:', tile, tiles_exc['20x'][tile])
        print('10x exc:', tile, tiles_exc['10x'][tile])
        print('5x inc:', tile, tiles_inc['5x'][tile])
        print('')
'''

'''
# consider tiles included in all mags
for tile in tiles_exc['40x']:
    
    tile_path = 'tiles'
    
    slidename = tile.split('_')[0]
    
    if tile not in tiles_exc['5x']:
        os.remove(f'{tile_path}/5x/{slidename}/{tile}.jpg')
        
    if tile not in tiles_exc['10x']:
        os.remove(f'{tile_path}/10x/{slidename}/{tile}.jpg')
        
    if tile not in tiles_exc['20x']:
        os.remove(f'{tile_path}/20x/{slidename}/{tile}.jpg')
'''

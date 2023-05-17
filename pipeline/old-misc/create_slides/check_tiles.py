import os

folders = ['tiles/5x', 'tiles/10x', 'tiles/20x', 'tiles/40x']

tiles_inc = {}

tiles_inc['all'] = {}

for path in folders:

    tiles_inc[path] = {}

    for slide in os.listdir(path):
        for tile in os.listdir(path+'/'+slide):

            if tile not in tiles_inc['all']:
                tiles_inc['all'][tile] = 1

            tiles_inc[path][tile] = 1

for tile in tiles_inc['all']:

    slide = tile.split('_')[0]

    notinc = False

    if (tile not in tiles_inc['tiles/5x'] or
        tile not in tiles_inc['tiles/10x'] or
        tile not in tiles_inc['tiles/20x'] or
        tile not in tiles_inc['tiles/40x']):
        notinc = True

    if notinc == True and tile in tiles_inc['tiles/5x']:
        os.remove(f'tiles/5x/{slide}/{tile}')

    if notinc == True and tile in tiles_inc['tiles/10x']:
       	os.remove(f'tiles/10x/{slide}/{tile}')

    if notinc == True and tile in tiles_inc['tiles/20x']:
       	os.remove(f'tiles/20x/{slide}/{tile}')

    if notinc == True and tile in tiles_inc['tiles/40x']:
       	os.remove(f'tiles/40x/{slide}/{tile}')

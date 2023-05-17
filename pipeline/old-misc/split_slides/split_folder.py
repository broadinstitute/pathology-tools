import os
import pandas as pd

path = '/workdir/sam4039/tiles/40x'

df = pd.read_csv('train_val_split.csv')

for _, row in df.iterrows():
    slide = row[0]
    split = row[1]

    for tile in os.listdir(f'{path}/{slide}'):
       os.rename(f'{path}/{slide}/{tile}',f'{path}/{split}/{tile}')

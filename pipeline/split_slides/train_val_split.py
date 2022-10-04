import pandas as pd
import os
from sklearn.model_selection import train_test_split

path = "/workdir/sam4039/tiles/10x"
slides=[] #slide names
y=[] #dummy
 
for slide in os.listdir(path):
    slides.append(slide)
    y.append(0)

slide_train,slide_val,y_train,y_val = train_test_split(slides,y,test_size=0.1,random_state=33)

slide_dict= {}

for slide in slides:
    if slide in slide_val:
        slide_dict[slide]="val"
    else:
        slide_dict[slide]="train"

slide_df = pd.DataFrame.from_dict(slide_dict,orient='index')
slide_df.to_csv('train_val_split.csv')


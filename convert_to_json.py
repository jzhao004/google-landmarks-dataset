import os 
import json 

import pandas as pd
import random
from tqdm import tqdm


ann_root = 'data/annotations'
train_pth = os.path.join(ann_root, 'train_clean_updated.csv')
save_to = os.path.join(ann_root, 'train.json')

random.seed(331712)

if __name__ == "__main__":
    train = pd.read_csv(train_pth)

    annotations = []
    
    # prepare dataset
    ## convert from df to list of dict 
    for i, row in tqdm(train.iterrows()):
        images = row['images'].split(' ')
        
        # remove samples with fewer than 6 images
        if len(images) < 6: 
            continue
            
        annotations.append({
            'id' : row['landmark_id'],
            'name' : row['page_title'],
            'images' : random.sample(images, 6),
            'summary' : row['summary'],
            'characteristics' : eval(row['characteristics']) if row['characteristics'] else None,
            'nearest_landmarks' : eval(row['nearest_landmark']) if row['nearest_landmark'] else None
            })             

        if (i!=0) & (i%10==0):
            with open(save_to, 'w') as f: 
                json.dump(annotations, f)
            
    with open(save_to, 'w') as f: 
        json.dump(annotations, f)
import os

import haversine.haversine as haversine
import numpy as np
import pandas as pd 
from tqdm import tqdm


ann_root = 'data/annotations'
train_pth = os.path.join(ann_root, 'train_clean_updated.csv')
save_to = os.path.join(ann_root, 'train_clean_updated.csv')

k = 10 # no. of nearest landmarks
d = 50 

if __name__ == "__main__":
    train = pd.read_csv(train_pth)
    n = len(train)
    print(f'No. of training samples: {n}')    

    coordinates = train[['lat', 'lon']].to_numpy()

    for i in tqdm(range(len(train))):
        # calculate haversine dist
        curr_coordinates = np.repeat(np.expand_dims(train.loc[i, ['lat', 'lon']], 0), n, axis=0)
        dist = np.array(list(map(haversine, curr_coordinates, coordinates)))
        
        # find top k neighbours 
        idx = np.array(sorted(range(len(dist)), key=lambda x: dist[x])[1:k+1])
        idx = [i for i in idx if dist[i] < d] # remove neighbours more than d km away 

        train.loc[i, 'nearest_landmark'] = str(list(train.loc[idx, 'page_title'])) if len(idx) > 0 else None
        train.loc[i, 'nearest_landmark_dist'] = str(list(np.around(dist[idx], 4))) if len(idx) > 0 else None
            
        if i%10 == 0:
            train.to_csv(save_to, index=False)
        
    train.to_csv(save_to, index=False)

import torch
from torch.utils.data import Dataset
import numpy as np

import pickle
import random

ATTRIBUTE = 'Attractiveness' # Trustworthiness, Dominance, Attractiveness

class WPlayDataset(Dataset):
    """W Play dataset."""

    #def __init__(self, w_file, attribute_file, score_dis_file, increment=None):
    def __init__(self, w_file, attribute_file, increment=None):
        
        self.w = np.load(w_file)

        with open(attribute_file,'rb') as f:
            self.attribute = pickle.load(f)

        #with open(score_dis_file, 'rb') as f:
        #    self.score_dis = pickle.load(f)

        self.increment = increment
        
        if ATTRIBUTE == 'Attractiveness':
            self.upper_limit = 0.73
        else:
            self.upper_limit = 0.62

    def __len__(self):
        return len(self.w)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        w = self.w[idx]
        #img_path, dis, score = self.attribute[idx]
        img_path, avg = self.attribute[idx]
        #print('IN WPlayDataset')
        
        # calculate target score
        '''
        tar_score = min(score+self.increment, 0.62) # 0.62 is the limit
        while (self.score_dis['%.2f' % tar_score] == []):
           tar_score += 0.01

        tar_dis = random.choice(self.score_dis['%.2f' % tar_score])


        sample = {'w': w, 'img_path': img_path, 'dis': dis, 'score': score, 'tar_score': tar_score, 'tar_dis': tar_dis}
        '''
        tar_score = min(avg.item()+self.increment, self.upper_limit) # IMPORTANT: Upper limit depends on available labels in image predictions (for input images)
        sample = {'w': w, 'img_path': img_path, 'avg': avg, 'tar_score': np.array(tar_score)}
        
        '''
        #print(avg)
        #max_value = max(avg)
        #idx_max_value = list(avg).index(max_value)
        idx_max_value = 6
        tar_score = min(avg.item(idx_max_value)+self.increment, 0.62) # 0.62 is the limit
        updated_tar_score = []
        reduction_score = self.increment / 6
        for idx in range(7):
            if idx != idx_max_value:
                updated_tar_score.append(avg.item(idx)-reduction_score)
            else:
                updated_tar_score.append(tar_score)
                
        #print(updated_tar_score)
        
        sample = {'w': w, 'img_path': img_path, 'avg': avg, 'tar_score': np.array(updated_tar_score)}
        '''
        
        return sample
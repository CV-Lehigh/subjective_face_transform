import torch
from torch.utils.data import Dataset
import numpy as np
import math

import pickle
import random

ATTRIBUTE = 'Attractiveness' # Trustworthiness, Dominance, Attractiveness

class WPlayDataset(Dataset):
    """W Play dataset."""

    def __init__(self, w_file, attribute_file, score_dis_file, increment=None):
        
        self.w = np.load(w_file)

        with open(attribute_file,'rb') as f:
            self.attribute = pickle.load(f)

        with open(score_dis_file, 'rb') as f:
            self.score_dis = pickle.load(f)

        self.increment = increment
        
        if ATTRIBUTE == 'Attractiveness':
            self.max_label = 5
        else:
            self.max_label = 7
            
        self.incr_decr_val = 0.0025
        self.method = '3' # NEW methods: 1, 2, 3

    def __len__(self):
        return len(self.w)
        
    def get_score(self, dis): # Weighted average
        score = np.dot(dis, np.arange(1, self.max_label+1))
        score = (score-1) / float(self.max_label-1)
        return score
        
    def get_max_value_pos_list(self, input_list):
        max_val = input_list[0]
        index_list = [0]
        for i in range(1, len(input_list)):
            if input_list[i] > max_val:
                max_val = input_list[i]
                index_list = []
                index_list.append(i)
            elif input_list[i] == max_val:
                index_list.append(i)
        return index_list
        
    def get_max_value_pos(self, input_list):
        max_val = input_list[0]
        index_val = 0
        for i in range(1, len(input_list)):
            if input_list[i] >= max_val:
                max_val = input_list[i]
                index_val = i
        return index_val

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        w = self.w[idx]
        img_path, dis, score = self.attribute[idx]

        # calculate target score
        #tar_score = min(score+self.increment, 0.62) # 0.62 is the limit
        tar_score = min(score+self.increment, 0.58) # IMPORTANT: Upper limit depends on available labels in image predictions
        #tar_score = min(score+self.increment, 0.75)
        #tar_score = min(score+self.increment, 1.00)
        
        ###while (self.score_dis['%.2f' % tar_score] == []):
        ###    tar_score += 0.01

        ###tar_dis = random.choice(self.score_dis['%.2f' % tar_score])
        
        if self.method == '1': # Use threshold to reduce or increase values in distribution
            dis_len = len(dis)
            thresh_pos = int(math.ceil(tar_score * dis_len))
            tar_dis = list(dis)
            
            incr_cnt = (self.max_label - thresh_pos) + 1
            incr_val = self.incr_decr_val / incr_cnt
            
            decr_cnt = self.max_label - incr_cnt
            decr_val = self.incr_decr_val / decr_cnt
            
            while round(self.get_score(np.array(tar_dis)), 4) != round(tar_score, 4):
                if any((item - decr_val) < 0.0 for item in tar_dis) or any((item + incr_val) > 1.0 for item in tar_dis):
                    break
                else:
                    count = 0
                    for val in tar_dis:
                        if count < (thresh_pos-1):
                            tar_dis[count] -= decr_val
                        else:
                            tar_dis[count] += incr_val
                        count += 1
                        
        if self.method == '2': # Find max probability value and increase it, while reducing other porbabilities in the same proportion
            tar_dis = list(dis)
            max_val_pos_list = self.get_max_value_pos_list(tar_dis)
            
            incr_cnt = len(max_val_pos_list)
            incr_val = self.incr_decr_val / incr_cnt
            
            decr_cnt = self.max_label - incr_cnt
            decr_val = self.incr_decr_val / decr_cnt
            
            while round(self.get_score(np.array(tar_dis)), 4) != round(tar_score, 4):
                if any((item - decr_val) < 0.0 for item in tar_dis) or any((item + incr_val) > 1.0 for item in tar_dis):
                    break
                else:
                    count = 0
                    for val in tar_dis:
                        if count not in max_val_pos_list:
                            tar_dis[count] -= decr_val
                        else:
                            tar_dis[count] += incr_val
                        count += 1
                        
        if self.method == '3': # Find max probability value and increase it, while reducing other porbabilities in the same proportion
            tar_dis = list(dis)
            max_val_pos = self.get_max_value_pos(tar_dis)
            
            if max_val_pos == 0:
                less_decr_list = [1]
            elif max_val_pos == self.max_label:
                less_decr_list = [self.max_label - 1]
            else:
                less_decr_list = [max_val_pos - 1, max_val_pos + 1]
            
            incr_val = self.incr_decr_val
            
            less_decr_val_tot = self.incr_decr_val / 4.0
            less_decr_val = less_decr_val_tot / len(less_decr_list)
            
            decr_cnt = self.max_label - len(less_decr_list) - 1
            decr_val = (self.incr_decr_val - less_decr_val_tot) / decr_cnt
            
            while round(self.get_score(np.array(tar_dis)), 4) != round(tar_score, 4):
                if any((item - decr_val) < 0.0 for item in tar_dis) or any((item + incr_val) > 1.0 for item in tar_dis):
                    break
                else:
                    count = 0
                    for val in tar_dis:
                        if count == max_val_pos:
                            tar_dis[count] += incr_val
                        elif count in less_decr_list:
                            tar_dis[count] -= less_decr_val
                        else:
                            tar_dis[count] -= decr_val
                        count += 1
        
        sample = {'w': w, 'img_path': img_path, 'dis': dis, 'score': score, 'tar_score': tar_score, 'tar_dis': np.array(tar_dis)}

        return sample
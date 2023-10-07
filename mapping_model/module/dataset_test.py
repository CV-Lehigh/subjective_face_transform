import torch
from torch.utils.data import Dataset
import numpy as np

import pickle
import random

class WTestDataset(Dataset):
    """W Play dataset."""

    def __init__(self, w_file, attribute_file):
        
        self.w = np.load(w_file)

        with open(attribute_file,'rb') as f:
            self.attribute = pickle.load(f)

    def __len__(self):
        return len(self.w)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        w = self.w[idx]
        img_path, dis, score = self.attribute[idx]


        sample = {'w': w, 'img_path': img_path, 'dis': dis, 'score': score}



        return sample
import torch
from torch.utils.data import Dataset
import numpy as np

class WDataset(Dataset):
    """W dataset."""

    def __init__(self, w_file, attribute_file):
        
        self.w = np.load(w_file)
        self.attribute = np.load(attribute_file)

    def __len__(self):
        return len(self.w)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        w = self.w[idx]
        attribute = self.attribute[idx]
        sample = {'w': w, 'attribute': attribute}

        return sample
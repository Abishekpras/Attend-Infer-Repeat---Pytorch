from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
import numpy as np
import torch

class MultiMNIST_Dataset(Dataset):

    def __init__(self, X, y, transform=None):
    
        self.mlb = MultiLabelBinarizer()
        self.transform = transform

        self.X = X
        self.y = self.mlb.fit_transform(y).astype(np.float32)

    def __getitem__(self, index):
        img = self.X[index]
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.from_numpy(self.y[index])
        return img, label

    def __len__(self):
        return len(self.X)
        

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    
    def __init__(self, train=True,subject_ids=[0]):
        self.data_path = "data/"
        self.data = []
        self.ind = None
        if train==True:    
            X = np.load("data/X_train_valid.npy")
            y = np.load("data/y_train_valid.npy")
            p = np.load("data/person_train_valid.npy").reshape(y.shape)
        else:    
            X = np.load("data/X_test.npy")
            y = np.load("data/y_test.npy")
            p = np.load("data/person_test.npy").reshape(y.shape)
        self.ind = np.zeros(y.shape)
        self.class_map = {769:0,770:1,771:2,772:3}
        for i in range(len(subject_ids)):
            self.ind = np.logical_or(self.ind,p==subject_ids[i])
        y = y[self.ind]
        X = X[self.ind]
        for i in range(y.shape[0]):
            self.data.append([X[i], y[i]])
        self.data_dim = (22, 1000)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item, class_name = self.data[idx]
        class_id = self.class_map[class_name]
        item_tensor = torch.from_numpy(item)
        class_id = torch.tensor(class_id)
        return item_tensor, class_id
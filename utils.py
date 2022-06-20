import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pdb

class MyDataset(Dataset):
    def __init__(self, x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, fold='train'):
        fold = fold.lower()

        self.train = False
        self.test = False
        self.val = False

        if fold == "train":
            self.train = True
        elif fold == "test":
            self.test = True
        elif fold == "val":
            self.val = True
        else:
            raise RuntimeError("Not train-val-test")

        self.x_train = x_train
        self.x_valid = x_valid
        self.x_test = x_test
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test
        self.id_train = id_train
        self.id_test = id_test

    def __len__(self):
        if self.train:
            return len(self.x_train)
        elif self.test:
            return len(self.x_test)
        elif self.val:
            return len(self.x_valid)

    def __getitem__(self, index):
        if self.train:
            x, y, cell_id = torch.from_numpy(np.array(self.x_train[index])).float(), \
                            torch.from_numpy(self.y_train[index]).float(), self.id_train[index]
        elif self.test:
            x, y, cell_id = torch.from_numpy(np.array(self.x_test[index])).float(), \
                            torch.from_numpy(self.y_test[index]).float(), self.id_test[index]
        elif self.val:
            x, y, cell_id = torch.from_numpy(np.array(self.x_valid[index])).float(), \
                            torch.from_numpy(self.y_valid[index]).float(), []

        return x, y, cell_id


    def collate(self, batches):
        xs = batches[0][0].unsqueeze(0)
        ys = batches[0][1].unsqueeze(0)
        ids = batches[0][2]
        return torch.FloatTensor(xs), torch.FloatTensor(ys), ids

import os
import torch
import numpy as np
import copy
from torch.utils.data import Dataset
import pdb
from tqdm import tqdm

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
        xs = torch.stack([batch[0] for batch in batches if len(batch) > 0])
        ys = torch.stack([batch[1] for batch in batches if len(batch) > 0])
        ids = [batch[2] for batch in batches if len(batch) > 0]
        return torch.FloatTensor(xs), torch.FloatTensor(ys), ids


def mixup(x, x_p, alpha=1.0):
    batch_size = min(x.shape[0], x_p.shape[0])
    lam = np.random.beta(alpha, alpha)
    # x = np.random.permutation(x)
    # x_p = np.random.permutation(x_p)
    x_mix = lam * x[:batch_size] + (1 - lam) * x_p[:batch_size]
    return x_mix, lam


def mixups(args, data, p_idx, labels_, cell_type):
    np.random.seed(args.seed * 2)
    data_augmented = copy.deepcopy(data)
    labels_augmented = copy.deepcopy(labels_)
    cell_type_augmented = copy.deepcopy(cell_type)
    # intra-mixup
    if args.intra_mixup is True:
        print("======= intra patient mixup ... ============")
        for idx, i in enumerate(p_idx):
            if len(i) < args.min_size:
                diff = 0
                sampled_idx_1 = []
                sampled_idx_2 = []
                for ct in set(cell_type[i]):
                    i_sub = i[cell_type[i] == ct]
                    diff_sub = max((args.min_size - len(i)) * len(i_sub) // len(i), 1)
                    sampled_idx_1 += np.random.choice(i_sub, diff_sub).tolist()
                    sampled_idx_2 += np.random.choice(i_sub, diff_sub).tolist()
                    cell_type_augmented = np.concatenate([cell_type_augmented, [ct] * diff_sub])
                    diff += diff_sub
                x_mix, lam = mixup(data[sampled_idx_1], data[sampled_idx_2], alpha=args.alpha)
                data_augmented = np.concatenate([data_augmented, x_mix])
                labels_augmented = np.concatenate([labels_augmented, [labels_augmented[i[0]]] * diff])
                p_idx[idx] = np.concatenate([i, np.arange(labels_augmented.shape[0] - diff, labels_augmented.shape[0])])

    if args.inter_only:
        p_idx_augmented = []
    else:
        p_idx_augmented = copy.deepcopy(p_idx)
        # inter-mixup
    if args.augment_num > 0:
        print("======= inter patient mixup ... ============")
        for i in tqdm(range(args.augment_num)):
            id_1, id_2 = np.random.randint(len(p_idx), size=2)
            idx_1, idx_2 = p_idx[id_1], p_idx[id_2]
            diff = 0
            sampled_idx_1 = []
            sampled_idx_2 = []
            set_intersection = set(cell_type_augmented[idx_1]).intersection(set(cell_type_augmented[idx_2]))
            while diff < args.min_size:
                for ct in set_intersection:
                    i_sub_1 = idx_1[cell_type_augmented[idx_1] == ct]
                    i_sub_2 = idx_2[cell_type_augmented[idx_2] == ct]
                    diff_sub = max(args.min_size * (len(i_sub_1)+len(i_sub_2)) // (len(idx_1)+len(idx_2)), 1)
                    sampled_idx_1 += np.random.choice(i_sub_1, diff_sub).tolist()
                    sampled_idx_2 += np.random.choice(i_sub_2, diff_sub).tolist()
                    cell_type_augmented = np.concatenate([cell_type_augmented, [ct] * diff_sub])
                    diff += diff_sub
            x_mix, lam = mixup(data_augmented[sampled_idx_1], data_augmented[sampled_idx_2], alpha=args.alpha)
            data_augmented = np.concatenate([data_augmented, x_mix])
            labels_augmented = np.concatenate([labels_augmented, [lam * labels_augmented[idx_1[0]] + (1 - lam) * labels_augmented[idx_2[0]]] * diff])
            p_idx_augmented.append(np.arange(labels_augmented.shape[0] - diff, labels_augmented.shape[0]))

    return data_augmented, p_idx_augmented, labels_augmented


def sampling(args, train_p_idx, test_p_idx, labels_):
    np.random.seed(args.seed * 3)
    if args.all == 0:
        individual_train = []
        individual_test = []
        for idx in train_p_idx:
            y = labels_[idx[0]]
            if idx.shape[0] < args.train_sample_cells:
                sample_cells = idx.shape[0] // 2
            else:
                sample_cells = args.train_sample_cells
            temp = []
            for _ in range(args.train_num_sample):
                sample = np.random.choice(idx, sample_cells, replace=False)
                temp.append((sample, y))
            individual_train.append(temp)
        for idx in test_p_idx:
            y = labels_[idx[0]]
            if idx.shape[0] < args.test_sample_cells:
                sample_cells = idx.shape[0] // 2
            else:
                sample_cells = args.test_sample_cells
            temp = []
            for _ in range(args.test_num_sample):
                sample = np.random.choice(idx, sample_cells, replace=False)
                temp.append((sample, y))
            individual_test.append(temp)
    else:
        individual_train = []
        individual_test = []
        for idx in train_p_idx:
            y = labels_[idx[0]]
            temp = []
            sample = idx
            temp.append((sample, y))
            individual_train.append(temp)
        for idx in test_p_idx:
            y = labels_[idx[0]]
            temp = []
            sample = idx
            temp.append((sample, y))
            individual_test.append(temp)

    return individual_train, individual_test

import numpy as np
from tqdm import tqdm
import torch
import pickle
# import cloudpred
import time
import pdb
import random
import scipy
from scipy.io import mmread
import scanpy as sc
from sklearn.decomposition import PCA
import scanpy as sc
import pandas as pd
from sklearn.model_selection import train_test_split


# def scRNA_data(cell_num=100):

#     X_train, X_test, X_valid = [], [], []
#     y_train, y_test, y_valid = [], [], []

#     filename = 'simu_data_merge.txt'
#     with open(filename, 'rb') as fp:
#         itemlist = pickle.load(fp)
#     for g, item in enumerate(itemlist):
#         _, train_data, _, test_data = item[0], item[1], item[2], item[3]
#         for data in train_data:
#             X_train.append(data)
#             if g < 1:
#                 y_train.append(0)
#             else:
#                 y_train.append(1)
#         for data in test_data:
#             X_test.append(data)
#             if g < 1:
#                 y_test.append(0)
#             else:
#                 y_test.append(1)

#     X_train, X_test, X_valid, y_train, y_test, y_valid = np.asarray(X_train), np.asarray(X_test), np.asarray(
#         X_valid), np.asarray(y_train), np.asarray(y_test), np.asarray(y_valid)

#     print("train / valid / test = ", X_train.shape, X_valid.shape, X_test.shape)

#     return X_train, X_valid, X_test, y_train.reshape([-1, 1]), y_valid, y_test.reshape([-1, 1])

# def Covid_data_1(args):
#     num_sample = 300
#     sample_cells = args.sample_cells
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     all = mmread(f'matrix.mtx').T.todense()

#     stats_df = pd.read_csv(f'GSE163005_annotation_patients.csv')

#     genes = []
#     idxx = np.arange(stats_df['x'].shape[0])
#     for i in set(stats_df['x']):
#         genes.append(all[idxx[stats_df['x'] == i]])

#     for i in range(len(genes)):
#         adata = sc.AnnData(genes[i])
#         sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
#         sc.pp.log1p(adata)
#         genes[i] = adata.X


#     # PCA
#     pca = PCA(n_components=args.dims, random_state=3456).fit(all)

#     for i in range(len(genes)):
#         genes[i] = pca.transform(genes[i])

#     Xtrain, Xvalid, Xtest = [], [], []
#     for i in range(len(genes)):
#         Xtrain.append((genes[i][:1000], i, None))
#         Xtest.append((genes[i][1000:], i, None))

#     Xtrain = list(map(lambda x: (x[0], *x[1:]), Xtrain))
#     Xvalid = list(map(lambda x: (x[0], *x[1:]), Xvalid))
#     Xtest = list(map(lambda x: (x[0], *x[1:]), Xtest))

#     X_train, X_test, X_valid = [], [], []
#     y_train, y_test, y_valid = [], [], []

#     for item in Xtrain:
#         for i in range(num_sample):
#             sample = np.random.choice(np.arange(item[0].shape[0]), sample_cells)
#             X_train.append(item[0][sample])
#             y_train.append(item[1])
#     for item in Xtest:
#         for i in range(num_sample):
#             sample = np.random.choice(np.arange(item[0].shape[0]), sample_cells)
#             X_test.append(item[0][sample])
#             y_test.append(item[1])
#     for item in Xvalid:
#         for i in range(num_sample):
#             sample = np.random.choice(np.arange(item[0].shape[0]), sample_cells)
#             X_valid.append(item[0][sample])
#             y_valid.append(item[1])

#     X_train, X_test, X_valid, y_train, y_test, y_valid = np.asarray(X_train), np.asarray(X_test), np.asarray(
#         X_valid), np.asarray(y_train), np.asarray(y_test), np.asarray(y_valid)

#     y_train = y_train.reshape([-1, 1])
#     y_test = y_test.reshape([-1, 1])

#     print("train / valid / test = ", X_train.shape, X_valid.shape, X_test.shape)

#     np.save('X_train', X_train)
#     np.save('X_test', X_test)
#     np.save('y_train', y_train)
#     np.save('y_test', y_test)

#     # X_train = np.load('X_train.npy')
#     # X_test = np.load('X_test.npy')
#     # y_train = np.load('y_train.npy')
#     # y_test = np.load('y_test.npy')


#     return X_train, X_train, X_test, y_train, y_train, y_test

# def Covid_data_1(args):
#     # num_sample = 300
#     # sample_cells = args.sample_cells
#     # random.seed(args.seed)
#     # np.random.seed(args.seed)
#     # torch.manual_seed(args.seed)
#     #
#     # genes_1 = mmread(f'GSE157526_RAW/S_1/matrix.mtx').T
#     # genes_2 = mmread(f'GSE157526_RAW/S_2/matrix.mtx').T
#     # genes_3 = mmread(f'GSE157526_RAW/S_3/matrix.mtx').T
#     # genes_4 = mmread(f'GSE157526_RAW/S_5/matrix.mtx').T
#     # genes_5 = mmread(f'GSE157526_RAW/S_6/matrix.mtx').T
#     #
#     # genes = [genes_1.todense(), genes_2.todense(), genes_3.todense(), genes_4.todense(), genes_5.todense()]
#     # all = None
#     #
#     # for i in range(len(genes)):
#     #     adata = sc.AnnData(genes[i])
#     #     sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
#     #     sc.pp.log1p(adata)
#     #     genes[i] = adata.X
#     #     if i == 0:
#     #         all = genes[i]
#     #     else:
#     #         all = np.concatenate((all, genes[i]), axis=0)
#     #
#     #
#     # # PCA
#     # pca = PCA(n_components=args.dims, random_state=3456).fit(all)
#     #
#     # for i in range(len(genes)):
#     #     genes[i] = pca.transform(genes[i])
#     #
#     # Xtrain, Xvalid, Xtest = [], [], []
#     # for i in range(len(genes)):
#     #     Xtrain.append((genes[i][:1000], i, None))
#     #     Xtest.append((genes[i][1000:], i, None))
#     #
#     # Xtrain = list(map(lambda x: (x[0], *x[1:]), Xtrain))
#     # Xvalid = list(map(lambda x: (x[0], *x[1:]), Xvalid))
#     # Xtest = list(map(lambda x: (x[0], *x[1:]), Xtest))
#     #
#     # X_train, X_test, X_valid = [], [], []
#     # y_train, y_test, y_valid = [], [], []
#     #
#     # for item in Xtrain:
#     #     for i in range(num_sample):
#     #         sample = np.random.choice(np.arange(item[0].shape[0]), sample_cells)
#     #         X_train.append(item[0][sample])
#     #         y_train.append(item[1])
#     # for item in Xtest:
#     #     for i in range(num_sample):
#     #         sample = np.random.choice(np.arange(item[0].shape[0]), sample_cells)
#     #         X_test.append(item[0][sample])
#     #         y_test.append(item[1])
#     # for item in Xvalid:
#     #     for i in range(num_sample):
#     #         sample = np.random.choice(np.arange(item[0].shape[0]), sample_cells)
#     #         X_valid.append(item[0][sample])
#     #         y_valid.append(item[1])
#     #
#     # X_train, X_test, X_valid, y_train, y_test, y_valid = np.asarray(X_train), np.asarray(X_test), np.asarray(
#     #     X_valid), np.asarray(y_train), np.asarray(y_test), np.asarray(y_valid)
#     #
#     # y_train = y_train.reshape([-1, 1])
#     # y_test = y_test.reshape([-1, 1])
#     #
#     # print("train / valid / test = ", X_train.shape, X_valid.shape, X_test.shape)
#     #
#     # np.save('X_train', X_train)
#     # np.save('X_test', X_test)
#     # np.save('y_train', y_train)
#     # np.save('y_test', y_test)

#     X_train = np.load('X_train.npy')
#     X_test = np.load('X_test.npy')
#     y_train = np.load('y_train.npy')
#     y_test = np.load('y_test.npy')


#     return X_train, X_train, X_test, y_train, y_train, y_test


# def Cloudpred_data(args):
#     num_sample = 10
#     sample_cells = args.sample_cells
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     t = time.time()
#     Xtrain, Xvalid, Xtest, state = cloudpred.utils.load_synthetic(args.dir, valid=args.valid, test=args.test,
#                                                                   train_patients=args.train_patients, cells=args.cells)


#     ### Transform data ###
#     t = time.time()
#     if args.transform == "none":
#         pass
#     elif args.transform == "log":
#         Xtrain = list(map(lambda x: (x[0].log1p(), *x[1:]), Xtrain))
#         Xvalid = list(map(lambda x: (x[0].log1p(), *x[1:]), Xvalid))
#         Xtest = list(map(lambda x: (x[0].log1p(), *x[1:]), Xtest))
#     else:
#         message = "Transform " + args.transform + " is not implemented."
#         raise NotImplementedError(message)

#     if args.pc:
#         iterations = 5
#         try:
#             pc = np.load(args.dir + "/pc_" + args.transform + "_" + str(args.seed) + "_" + str(args.dims) + "_" + str(
#                 iterations) + ".npz")["pc"]
#         except FileNotFoundError:
#             pc = cloudpred.utils.train_pca_autoencoder(scipy.sparse.vstack(map(lambda x: x[0], Xtrain)), None,
#                                                        scipy.sparse.vstack(map(lambda x: x[0], Xvalid)), None,
#                                                        args.dims, args.transform,
#                                                        iterations=iterations,
#                                                        figroot=args.figroot)  # TODO: get rid of figroot?
#             np.savez_compressed(
#                 args.dir + "/pc_" + args.transform + "_" + str(args.seed) + "_" + str(args.dims) + "_" + str(
#                     iterations) + ".npz",
#                 pc=pc)

#         pc = pc[:, :args.dims]

#         ### Project onto principal components ###
#         mu = scipy.sparse.vstack(list(map(lambda x: x[0], Xtrain))).mean(axis=0)
#         Xtrain = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xtrain))  # - np.asarray(mu.dot(pc))
#         Xvalid = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xvalid))  # - np.asarray(mu.dot(pc))
#         Xtest = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xtest))  # - np.asarray(mu.dot(pc))
#         full = np.concatenate(list(map(lambda x: x[0], Xtrain)))
#         mu = np.mean(full, axis=0)
#         sigma = np.sqrt(np.mean(np.square(full - mu), axis=0))
#         sigma = sigma[0, 0]
#         Xtrain = list(map(lambda x: ((x[0] - mu) / sigma, *x[1:]), Xtrain))  # - np.asarray(mu.dot(pc))
#         Xvalid = list(map(lambda x: ((x[0] - mu) / sigma, *x[1:]), Xvalid))  # - np.asarray(mu.dot(pc))
#         Xtest = list(map(lambda x: ((x[0] - mu) / sigma, *x[1:]), Xtest))  # - np.asarray(mu.dot(pc))
#     else:
#         Xtrain = list(map(lambda x: (x[0].astype(np.float32).todense(), *x[1:]), Xtrain))
#         Xvalid = list(map(lambda x: (x[0].astype(np.float32).todense(), *x[1:]), Xvalid))
#         Xtest = list(map(lambda x: (x[0].astype(np.float32).todense(), *x[1:]), Xtest))

#     X_train, X_test, X_valid = [], [], []
#     y_train, y_test, y_valid = [], [], []

#     if (args.model == 'linear') or (sample_cells == 1000):
#         for item in Xtrain:
#             X_train.append(item[0])
#             y_train.append(item[1])
#         for item in Xtest:
#             X_test.append(item[0])
#             y_test.append(item[1])
#         for item in Xvalid:
#             X_valid.append(item[0])
#             y_valid.append(item[1])
#     else:
#         for item in Xtrain:
#             for i in range(num_sample):
#                 sample = np.random.choice(np.arange(item[0].shape[0]), sample_cells)
#                 X_train.append(item[0][sample])
#                 y_train.append(item[1])
#         for item in Xtest:
#             for i in range(num_sample):
#                 sample = np.random.choice(np.arange(item[0].shape[0]), sample_cells)
#                 X_test.append(item[0][sample])
#                 y_test.append(item[1])
#         for item in Xvalid:
#             for i in range(num_sample):
#                 sample = np.random.choice(np.arange(item[0].shape[0]), sample_cells)
#                 X_valid.append(item[0][sample])
#                 y_valid.append(item[1])

#     X_train, X_test, X_valid, y_train, y_test, y_valid = np.asarray(X_train), np.asarray(X_test), np.asarray(
#         X_valid), np.asarray(y_train), np.asarray(y_test), np.asarray(y_valid)

#     print("train / valid / test = ", X_train.shape, X_valid.shape, X_test.shape)

#     return X_train, X_valid, X_test, y_train.reshape([-1, 1]), y_valid, y_test.reshape([-1, 1])


# def Covid_data(args):
#     a_file = open(args.train_dataset, "rb")
#     data = pickle.load(a_file)
#     a_file.close()
#     X_train, y_train, id_train = data['X_train'], data['y_train'], data['id_train']
#
#     a_file = open(args.test_dataset, "rb")
#     data = pickle.load(a_file)
#     a_file.close()
#     X_test, y_test, id_test = data['X_test'], data['y_test'],  data['id_test']
#
#     num_train = X_train.shape[0]
#     random_idx = np.random.permutation(np.arange(num_train))
#     X_valid, X_train = X_train[random_idx[:num_train // 10]], X_train[random_idx[num_train // 10:]]
#     y_valid, y_train = y_train[random_idx[:num_train // 10]], y_train[random_idx[num_train // 10:]]
#
#     print("train / valid / test = ", X_train.shape, X_valid.shape, X_test.shape)
#
#     return X_train, X_valid, X_test, y_train.reshape([-1, 1]), y_valid.reshape([-1, 1]), y_test.reshape([-1, 1]), id_train, id_test


def Covid_data(args):
    random.seed(args.seed+1)
    np.random.seed(args.seed+2)

    with open('covid_pca.npy', 'rb') as f:
        data = np.load(f)

    a_file = open('patient_id.pkl', "rb")
    patient_id = pickle.load(a_file)
    a_file.close()

    a_file = open(args.task+'_label.pkl', "rb")
    labels = pickle.load(a_file)
    a_file.close()

    indices = np.arange(data.shape[0])
    p_ids = list(set(patient_id))
    p_idx = []
    for i in p_ids:
        p_idx.append(indices[patient_id == i])

    id_dict = {}
    if args.task == 'severity':
        id_dict = {'mild/moderate': 0, 'severe/critical': 1}
    elif args.task == 'sex':
        id_dict = {'M': 0, 'F': 1}
    elif args.task == 'stage':
        id_dict = {'convalescence': 0, 'progression': 1}

    individual_train = []
    individual_test = []
    for idx in p_idx:
        y = id_dict.get(labels[idx[0]], -1)
        if y == -1:
            continue

        if idx.shape[0] < args.train_sample_cells:
            sample_cells = idx.shape[0] // 2
        else:
            sample_cells = args.train_sample_cells
        temp = []
        for _ in range(args.train_num_sample):
            sample = np.random.choice(idx, sample_cells, replace=False)
            temp.append((sample, y))
        individual_train.append(temp)

        if idx.shape[0] < args.test_sample_cells:
            sample_cells = idx.shape[0] // 2
        else:
            sample_cells = args.test_sample_cells
        temp = []
        for _ in range(args.test_num_sample):
            sample = np.random.choice(idx, sample_cells, replace=False)
            temp.append((sample, y))
        individual_test.append(temp)

    return data, individual_train, individual_test



    # X_train, y_train, id_train = data['X_train'], data['y_train'], data['id_train']
    #
    # a_file = open(args.test_dataset, "rb")
    # data = pickle.load(a_file)
    # a_file.close()
    # X_test, y_test, id_test = data['X_test'], data['y_test'],  data['id_test']
    #
    # num_train = X_train.shape[0]
    # random_idx = np.random.permutation(np.arange(num_train))
    # X_valid, X_train = X_train[random_idx[:num_train // 10]], X_train[random_idx[num_train // 10:]]
    # y_valid, y_train = y_train[random_idx[:num_train // 10]], y_train[random_idx[num_train // 10:]]
    #
    # random_idx = np.random.permutation(np.arange(2000))
    # X_test1 = np.concatenate([X_test[random_idx[:1000]], X_train[:900]], 0)
    # X_train1 = np.concatenate([X_test[random_idx[1000:]], X_train[900:]], 0)
    # y_test = np.concatenate([np.zeros(1000), np.ones(900)], 0)
    # y_train = np.concatenate([np.zeros(1000), np.ones(900)], 0)
    #
    #
    #
    # print("train / valid / test = ", X_train1.shape, X_valid.shape, X_test1.shape)
    #
    # return X_train1, X_valid, X_test1, y_train.reshape([-1, 1]), y_valid.reshape([-1, 1]), y_test.reshape([-1, 1]), id_train, id_test

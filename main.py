import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import *
import os
import pandas as pd
import argparse
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import RepeatedKFold
import json
from tqdm import tqdm
import collections
import pdb

from model_baseline import *
# from model_informer import Informer
# from model_longformer import Longformer
# from model_reformer import Reformer
import matplotlib.pyplot as plt

from dataloader import *

def _str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")
def int_or_float(x):
    try:
        return int(x)
    except ValueError:
        return float(x)

parser = argparse.ArgumentParser(description='scRNA diagnosis')

parser.add_argument('--seed', type=int, default=240)

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=50)

parser.add_argument("--dir", type=str, default="covid_data_stage_3_1000.pkl", help="root directory of data")

parser.add_argument("--train_dataset", type=str, default="covid_data_sex_4.pkl")
parser.add_argument("--test_dataset", type=str, default="covid_data_sex_4.pkl")

parser.add_argument("--task", type=str, default="severity")

parser.add_argument('--h_dim', type=int, default=128)  # hidden dim of the model
parser.add_argument('--dropout', type=float, default=0.3)  # dropout

parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument("-t", "--transform", default="log",
                        choices=["none", "log"],
                        help="preprocessing on data")
parser.add_argument("--pc", type=_str2bool, default=True,
                        help="project onto principal components")
parser.add_argument("--valid", type=int_or_float, default=0.25,
                        help="root for optional figures")
parser.add_argument("--test", type=int_or_float, default=0.25,
                        help="root for optional figures")
parser.add_argument("--train_patients", type=int, default=None,
                        help="limit number of training patients")
parser.add_argument("--cells", type=int, default=None,
                        help="limit number of cells")
parser.add_argument("--pca", type=int, default=50,
                        help="dimension of principal components")
parser.add_argument("-f", "--figroot", type=str, default=None,
                        help="root for optional figures")
parser.add_argument("--sample_cells", type=int, default=500,
                        help="number of cells in one sample")
parser.add_argument("--train_sample_cells", type=int, default=500,
                        help="number of cells in one sample in train dataset")
parser.add_argument("--test_sample_cells", type=int, default=500,
                        help="number of cells in one sample in test dataset")
parser.add_argument("--num_sample", type=int, default=1000,
                        help="number of sampled data points")
parser.add_argument("--train_num_sample", type=int, default=20,
                        help="number of sampled data points in train dataset")
parser.add_argument("--test_num_sample", type=int, default=100,
                        help="number of sampled data points in test dataset")

parser.add_argument('--model', type=str, default='Transformer')
parser.add_argument('--dataset', type=str, default=None)

parser.add_argument('--intra_mixup', type=_str2bool, default=True)
parser.add_argument('--augment_num', type=int, default=100)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--repeat', type=int, default=3)
parser.add_argument('--all', type=int, default=0)
parser.add_argument('--min_size', type=int, default=1000)


args = parser.parse_args()

# print("# of GPUs is", torch.cuda.device_count())
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# x_train, x_valid, x_test, y_train, y_valid, y_test = scRNA_data(cell_num=100)  # numpy array
# x_train, x_valid, x_test, y_train, y_valid, y_test = Cloudpred_data(args)
# x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test = Covid_data(args)


def train(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test):
    # x_train, x_valid, x_test, y_train, y_valid, y_test = \
    #     torch.from_numpy(x_train).float(), torch.from_numpy(x_valid).float(), torch.from_numpy(x_test).float(), \
    #     torch.from_numpy(y_train).long(), torch.from_numpy(y_valid).long(), torch.from_numpy(y_test).long()
    #
    #
    # train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)
    # valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, y_valid), batch_size=args.batch_size, shuffle=False)

    #
    #
    dataset_1 = MyDataset(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, fold='train')
    dataset_2 = MyDataset(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, fold='test')
    dataset_3 = MyDataset(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, fold='val')
    train_loader = torch.utils.data.DataLoader(dataset_1, batch_size=args.batch_size, shuffle=True, collate_fn=dataset_1.collate)
    test_loader = torch.utils.data.DataLoader(dataset_2, batch_size=1, shuffle=False, collate_fn=dataset_2.collate)
    valid_loader = torch.utils.data.DataLoader(dataset_3, batch_size=args.batch_size, shuffle=False, collate_fn=dataset_3.collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = 100
    input_dim = x_train[0].shape[-1]
    output_class = 1

    if args.model == 'Transformer':
        model = Transformer(seq_len=args.sample_cells, input_dim= input_dim, PCA_dim=args.pca, h_dim=args.h_dim, N=args.layers, heads=args.heads, dropout=args.dropout, cl=output_class)
    elif args.model == 'feedforward':
        model = FeedForward_(PCA_dim=args.pca, cl=output_class, dropout=args.dropout)
    elif args.model == 'linear':
        model = Linear_Classfier(PCA_dim=args.pca, cl=output_class)


    model = nn.DataParallel(model)
    model.to(device)

    print(device)

    stats = {}

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)
    sigmoid = torch.nn.Sigmoid().to(device)

    max_acc, max_epoch, max_auc, max_loss = 0, 0, 0, 0
    test_accs, valid_accs, train_losses, train_accs = [], [], [], []
    for ep in range(1, args.epochs+1):
        model.train()
        train_loss = []

        #pred = []
        #true = []

        for batch in tqdm(train_loader):
            x_ = batch[0].to(device)
            y_ = batch[1].to(device)

            optimizer.zero_grad()

            out = model(x_)

            # if y_ != 1 and y_ != 0:
            #     print(sigmoid(out), y_)

            loss = nn.BCELoss()(sigmoid(out), y_)
            loss.backward()

            optimizer.step()
            train_loss.append(loss.item())

            #out = sigmoid(out)
            #out = out.detach().cpu().numpy()

            #pred.append(out)
            #y_ = y_.detach().cpu().numpy()
            #true.append(y_)

        # scheduler.step()


        #pred = np.concatenate(pred)
        #true = np.concatenate(true)
        #fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
        #AUC = metrics.auc(fpr, tpr)

        if ep % 1 == 0:
            model.eval()
            pred = []
            true = []
            with torch.no_grad():
                for batch in (test_loader):
                    x_ = batch[0].to(device).squeeze(0)
                    y_ = batch[1].to(device)

                    out = model(x_)
                    out = sigmoid(out)
                    out = out.detach().cpu().numpy()

                    # attens = model.module.attens
                    # topK = np.bincount(attens.max(-1)[1].cpu().detach().numpy().reshape(-1)).argsort()[-20:][::-1]
                    # for types in id_[topK]:
                    #     if stats.get(types, 0) == 0:
                    #         stats[types] = 1
                    #     else:
                    #         stats[types] += 1

                    # majority voting
                    f = lambda x: 1 if x >= 0.5 else 0
                    func = np.vectorize(f)
                    out = np.argmax(np.bincount(func(out).reshape(-1))).reshape(-1)
                    pred.append(out)
                    y_ = y_.detach().cpu().numpy()
                    true.append(y_)
            pred = np.concatenate(pred)
            true = np.concatenate(true).reshape(-1)


            train_loss = sum(train_loss) / len(train_loss)
            train_losses.append(train_loss)
            train_acc = 0
            valid_acc = 0

            # fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
            # test_AUC = metrics.auc(fpr, tpr)
            test_AUC = 0

            test_acc = accuracy_score(true, pred)

            test_accs.append(test_acc)


            # pred = []
            # true = []
            # with torch.no_grad():
            #     for batch in tqdm(train_loader):
            #         x_ = batch[0].to(device)
            #         y_ = batch[1].to(device)
            #
            #         out = model(x_)
            #         out = sigmoid(out)
            #         out = out.detach().cpu().numpy()
            #
            #         # attens = model.module.attens
            #         # topK = np.bincount(attens.max(-1)[1].cpu().detach().numpy().reshape(-1)).argsort()[-20:][::-1]
            #         # for types in id_[topK]:
            #         #     if stats.get(types, 0) == 0:
            #         #         stats[types] = 1
            #         #     else:
            #         #         stats[types] += 1
            #
            #         pred.append(out)
            #         y_ = y_.detach().cpu().numpy()
            #         true.append(y_)
            # pred = np.concatenate(pred)
            # true = np.concatenate(true)
            # pred = pred.argmax(1)
            # train_acc = accuracy_score(true.reshape(-1), pred)
            # train_accs.append(train_acc)

            if test_acc >= max_acc:
                max_epoch = ep
                max_acc = test_acc
                max_loss = train_loss

            #print("Epoch %d, Train Loss %f, Train AUC %f Test AUC %f,"%(ep, train_loss, AUC, test_AUC))
            # print("Epoch %d, Train Loss %f, Train ACC %f, Valid ACC %f, Test ACC %f,"%(ep, train_loss, train_acc, valid_acc, test_acc))
            print("Epoch %d, Train Loss %f, Test ACC %f,"%(ep, train_loss, test_acc))

        # print(stats)
    print("Best performance: Epoch %d, Loss %f, Test ACC %f," % (max_epoch, max_loss, max_acc))


    #####################
    # Visualization
    #####################
    # start_epoch = 0
    # end_epoch = args.epochs
    # plot_num = 1
    # label = [None] * plot_num
    # label[0] = 'method=0, LR=0.001'
    # fig = make_subplots(rows=2, cols=2)
    # colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    # col_num = len(colors)
    #
    # epoch = np.arange(start_epoch, end_epoch)
    #
    # def plot_sub(sub_val, sub_row, sub_col, sub_x_title, sub_y_title, sub_showlegend, log_y=True):
    #     linewidth = 1
    #     dash_val_counter = 0
    #     dash_val = None
    #
    #     for k in range(plot_num):
    #         fig.add_trace(go.Scatter(x=epoch, y=sub_val[k][start_epoch:end_epoch],
    #                                  name=label[k], line=dict(color=colors[k % col_num], width=linewidth, dash=dash_val),
    #                                  legendgroup=str(k), showlegend=sub_showlegend),
    #                       row=sub_row, col=sub_col)
    #         if dash_val_counter == 0:
    #             dash_val = 'dashdot'
    #         elif dash_val_counter == 1:
    #             dash_val = 'dash'
    #         else:
    #             dash_val = None
    #             dash_val_counter = -1
    #         dash_val_counter += 1
    #     fig.update_xaxes(title_text=sub_x_title, row=sub_row, col=sub_col)
    #     if log_y == True:
    #         fig.update_yaxes(title_text=sub_y_title, row=sub_row, col=sub_col, type="log")
    #     else:
    #         fig.update_yaxes(title_text=sub_y_title, row=sub_row, col=sub_col)
    #
    #
    # plot_sub([train_losses], 1, 1, 'epoch', 'train loss', True, log_y=True)
    # plot_sub([train_accs], 1, 2, 'epoch', 'train acc', False, log_y=True)
    # plot_sub([valid_accs], 2, 1, 'epoch', 'valid acc', False, log_y=True)
    # plot_sub([test_accs], 2, 2, 'epoch', 'test acc', False, log_y=True)
    #
    # if not os.path.exists(args.dir):
    #     os.makedirs(args.dir)
    #
    # fig.write_html(args.dir + '/' + args.train_dataset[24:-4] + '_' + args.test_dataset[30:-4] + '.html')

    return max_acc


data, p_idx, labels_ = Covid_data(args)
rkf = RepeatedKFold(n_splits=5, n_repeats=args.repeat, random_state=args.seed+3)
num = np.arange(len(p_idx))
accuracy = []
# TODO two methods: one is using batch-size=1, another is uisng batch-size=num_samples
# TODO the first method keeps this veriosn; for the second method, use the one in test_index

for train_index, test_index in rkf.split(num):
    x_train = []
    x_test = []
    x_valid = []
    y_train = []
    y_valid = []
    y_test = []
    id_train = []
    id_test = []
    # only use the augmented data (intra-mixup, inter-mixup) as train data
    data_augmented, train_p_idx, labels_augmented = mixups(args, data, [p_idx[idx] for idx in train_index], labels_)
    individual_train, individual_test = sampling(args, train_p_idx, [p_idx[idx] for idx in test_index], labels_augmented)
    for t in individual_train:
        id, label = [id_l[0] for id_l in t], [id_l[1] for id_l in t]
        x_train += [data_augmented[ii] for ii in id]
        y_train += (label)
        id_train += (id)
    for t in individual_test:
        id, label = [id_l[0] for id_l in t], [id_l[1] for id_l in t]
        x_test.append([data_augmented[ii] for ii in id])
        y_test.append(label[0])
        id_test.append(id)

    x_train, x_valid, x_test, y_train, y_valid, y_test = x_train, [],  x_test, np.array(y_train).reshape([-1, 1]), \
                                                        np.array([]).reshape([-1, 1]), np.array(y_test).reshape([-1, 1])

    acc = train(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test)
    accuracy.append(acc)


print("Best performance: Test ACC %f," % (np.average(accuracy)))

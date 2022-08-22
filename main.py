import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
import scipy.stats as st
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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
import copy
import json
from tqdm import tqdm
import collections
import pdb
from datetime import datetime

from model_baseline import *
from Transformer import TransformerPredictor
import transformers
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
parser.add_argument('--learning_rate', type=float, default=3e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=120)

parser.add_argument("--dir", type=str, default="covid_data", help="root directory of data")

parser.add_argument("--train_dataset", type=str, default="covid_data_sex_4.pkl")
parser.add_argument("--test_dataset", type=str, default="covid_data_sex_4.pkl")

parser.add_argument("--task", type=str, default="severity")

parser.add_argument('--emb_dim', type=int, default=128)  # embedding dim
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

parser.add_argument('--intra_mixup', type=_str2bool, default=False)
parser.add_argument('--inter_only', type=_str2bool, default=False)
parser.add_argument('--same_pheno', type=int, default=0)
parser.add_argument('--augment_num', type=int, default=0)
parser.add_argument('--alpha', type=float, default=1.0)  # TODO need to change
parser.add_argument('--repeat', type=int, default=3)
parser.add_argument('--all', type=int, default=1)
parser.add_argument('--min_size', type=int, default=6000)
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--pca', type=_str2bool, default=True)
parser.add_argument('--mix_type', type=int, default=1)
parser.add_argument('--intra_only', type=_str2bool, default=False)
parser.add_argument('--norm_first', type=_str2bool, default=False)
parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--warmup', type=_str2bool, default=False)

args = parser.parse_args()

# print("# of GPUs is", torch.cuda.device_count())
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

patient_summary = {}
stats = {}


# x_train, x_valid, x_test, y_train, y_valid, y_test = scRNA_data(cell_num=100)  # numpy array
# x_train, x_valid, x_test, y_train, y_valid, y_test = Cloudpred_data(args)
# x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test = Covid_data(args)


def train(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, data_augmented, data):
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
    train_loader = torch.utils.data.DataLoader(dataset_1, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=dataset_1.collate)
    test_loader = torch.utils.data.DataLoader(dataset_2, batch_size=1, shuffle=False, collate_fn=dataset_2.collate)
    valid_loader = torch.utils.data.DataLoader(dataset_3, batch_size=1, shuffle=False, collate_fn=dataset_3.collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = data_augmented[0].shape[-1]
    output_class = 1

    if args.model == 'Transformer':
        # model = Transformer(seq_len=args.sample_cells, input_dim=input_dim, emb_dim=args.emb_dim, h_dim=args.h_dim,
        #                     N=args.layers, heads=args.heads, dropout=args.dropout, cl=output_class, pca=args.pca)
        model = TransformerPredictor(input_dim=input_dim, model_dim=args.emb_dim, num_classes=output_class,
                                     num_heads=args.heads, num_layers=args.layers, dropout=args.dropout,
                                     input_dropout=0, pca=args.pca, norm_first=args.norm_first)
    elif args.model == 'feedforward':
        model = FeedForward(input_dim=input_dim, h_dim=args.h_dim, cl=output_class, dropout=args.dropout)
    elif args.model == 'linear':
        model = Linear_Classfier(input_dim=input_dim, cl=output_class)

    model = nn.DataParallel(model)
    model.to(device)
    best_model = model

    print(device)

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.warmup:
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.epochs // 10,
                                                                 num_training_steps=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, last_epoch=-1)
    sigmoid = torch.nn.Sigmoid().to(device)

    max_acc, max_epoch, max_auc, max_loss, max_valid_acc, max_valid_auc = 0, 0, 0, 0, 0, 0
    test_accs, valid_aucs, train_losses, valid_losses, train_accs, test_aucs = [], [0.], [], [], [], []
    best_valid_loss = float("inf")
    wrongs = []
    trigger_times = 0
    patience = 2
    for ep in (range(1, args.epochs + 1)):
        model.train()
        train_loss = []

        # pred = []
        # true = []

        for batch in (train_loader):
            x_ = torch.from_numpy(data_augmented[batch[0]]).float().to(device)
            y_ = batch[1].to(device)
            mask_ = batch[3].to(device)

            optimizer.zero_grad()

            out = model(x_, mask_)

            loss = nn.BCELoss()(sigmoid(out), y_)
            loss.backward()

            optimizer.step()
            train_loss.append(loss.item())

            # out = sigmoid(out)
            # out = out.detach().cpu().numpy()

            # pred.append(out)
            # y_ = y_.detach().cpu().numpy()
            # true.append(y_)

        scheduler.step()

        # my_lr = scheduler.get_lr()
        # print(my_lr)

        train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(train_loss)
        train_acc = 0

        # pred = np.concatenate(pred)
        # true = np.concatenate(true)
        # fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
        # AUC = metrics.auc(fpr, tpr)

        if ep % 1 == 0:
            valid_loss = []
            model.eval()
            pred = []
            true = []
            with torch.no_grad():
                for batch in (valid_loader):
                    x_ = torch.from_numpy(data[batch[0]]).float().to(device).squeeze(0)
                    y_ = batch[1].int().to(device)

                    out = model(x_)
                    out = sigmoid(out)
                    # out = out.detach().cpu().numpy()

                    loss = nn.BCELoss()(out, y_ * torch.ones(out.shape).to(device))
                    valid_loss.append(loss.item())

                    out = out.detach().cpu().numpy()

                    # attens = model.module.attens
                    # topK = np.bincount(attens.max(-1)[1].cpu().detach().numpy().reshape(-1)).argsort()[-20:][::-1]
                    # for types in id_[topK]:
                    #     if stats.get(types, 0) == 0:
                    #         stats[types] = 1
                    #     else:
                    #         stats[types] += 1

                    # majority voting
                    f = lambda x: 1 if x > 0.5 else 0
                    func = np.vectorize(f)
                    out = np.argmax(np.bincount(func(out).reshape(-1))).reshape(-1)
                    pred.append(out)
                    y_ = y_.detach().cpu().numpy()
                    true.append(y_)
            pred = np.concatenate(pred)
            true = np.concatenate(true)

            # fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
            # valid_auc = metrics.auc(fpr, tpr)
            valid_auc = metrics.roc_auc_score(true, pred)
            valid_acc = accuracy_score(true.reshape(-1), pred)
            valid_aucs.append(valid_auc)
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_losses.append(valid_loss)

            # if (valid_auc > max_valid_auc) and (ep >= args.threshold):
            if (valid_loss < best_valid_loss) and (ep >= args.threshold):
                best_model = copy.deepcopy(model)
                max_valid_auc = valid_auc
                max_epoch = ep
                best_valid_loss = valid_loss
                max_loss = train_loss

            print("Epoch %d, Train Loss %f, Valid_loss %f, Valid Auc %f" % (ep, train_loss, valid_loss, valid_auc))

            # if (ep > args.epochs - 50) and (valid_auc < valid_aucs[-2]):
            if (ep > args.epochs - 50) and ep > 1 and (valid_loss > valid_losses[-2]):
                trigger_times += 1
                if trigger_times >= patience:
                    break
            else:
                trigger_times = 0

    best_model.eval()
    pred = []
    true = []
    wrong = []
    with torch.no_grad():
        for batch in (test_loader):
            x_ = torch.from_numpy(data[batch[0]]).float().to(device).squeeze(0)
            y_ = batch[1].int().to(device)
            id_ = batch[2][0]

            out = best_model(x_)
            out = sigmoid(out)
            out = out.detach().cpu().numpy()

            if args.model == 'Transformer':
                attens = best_model.module.get_attention_maps(x_)[-1]
                for iter in range(len(attens)):
                    topK = np.bincount(attens[iter].max(-1)[1].cpu().detach().numpy().reshape(-1)).argsort()[-20:][::-1]
                    for types in cell_type_64[id_[iter][topK]]:
                        stats[types] = stats.get(types, 0) + 1

            # majority voting
            f = lambda x: 1 if x > 0.5 else 0
            func = np.vectorize(f)
            out = np.argmax(np.bincount(func(out).reshape(-1))).reshape(-1)
            pred.append(out)
            y_ = y_.detach().cpu().numpy()
            true.append(y_)
            if out[0] != y_[0][0]:
                wrong.append(patient_id[batch[2][0][0][0]])

    pred = np.concatenate(pred)
    true = np.concatenate(true)
    if len(wrongs) == 0:
        wrongs = set(wrong)
    else:
        wrongs = wrongs.intersection(set(wrong))

    # fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
    # test_auc = metrics.auc(fpr, tpr)
    test_auc = metrics.roc_auc_score(true, pred)
    test_aucs.append(test_auc)

    test_acc = accuracy_score(true.reshape(-1), pred)
    test_accs.append(test_acc)

    cm = confusion_matrix(true.reshape(-1), pred).ravel()
    recall = cm[3] / (cm[3] + cm[2])
    precision = cm[3] / (cm[3] + cm[1])

    # print("Epoch %d, Train Loss %f, Train ACC %f, Valid ACC %f, Test ACC %f,"%(ep, train_loss, train_acc, valid_acc, test_acc))
    # print("Epoch %d, Train Loss %f, Test ACC %f,"%(ep, train_loss, test_acc))

    print("Best performance: Epoch %d, Loss %f, Test ACC %f, Test AUC %f, Test Recall %f, Test Precision %f" % (max_epoch, max_loss, test_acc, test_auc, recall, precision))
    print("Confusion Matrix: " +str(cm))
    for w in wrongs:
        v = patient_summary.get(w, 0)
        patient_summary[w] = v + 1

    ####################
    # Visualization
    ####################
    start_epoch = 0
    end_epoch = args.epochs
    plot_num = 1
    label = [None] * plot_num
    label[0] = 'method=0, LR=0.001'
    fig = make_subplots(rows=1, cols=2)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    col_num = len(colors)

    epoch = np.arange(start_epoch, end_epoch)

    def plot_sub(sub_val, sub_row, sub_col, sub_x_title, sub_y_title, sub_showlegend, log_y=True):
        linewidth = 1
        dash_val_counter = 0
        dash_val = None

        for k in range(plot_num):
            fig.add_trace(go.Scatter(x=epoch, y=sub_val[k][start_epoch:end_epoch],
                                     name=label[k],
                                     line=dict(color=colors[k % col_num], width=linewidth, dash=dash_val),
                                     legendgroup=str(k), showlegend=sub_showlegend),
                          row=sub_row, col=sub_col)
            if dash_val_counter == 0:
                dash_val = 'dashdot'
            elif dash_val_counter == 1:
                dash_val = 'dash'
            else:
                dash_val = None
                dash_val_counter = -1
            dash_val_counter += 1
        fig.update_xaxes(title_text=sub_x_title, row=sub_row, col=sub_col)
        if log_y == True:
            fig.update_yaxes(title_text=sub_y_title, row=sub_row, col=sub_col, type="log")
        else:
            fig.update_yaxes(title_text=sub_y_title, row=sub_row, col=sub_col)

    # plot_sub([train_losses], 1, 1, 'epoch', 'train loss', True, log_y=True)
    # plot_sub([train_accs], 1, 2, 'epoch', 'train acc', False, log_y=True)
    # plot_sub([valid_losses], 1, 2, 'epoch', 'valid loss', False, log_y=True)
    # plot_sub([test_accs], 2, 2, 'epoch', 'test acc', False, log_y=True)

    # if not os.path.exists(args.dir):
    #     os.makedirs(args.dir)

    # fig.write_html(args.dir + '/' + str(iter_count) + '_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.html')

    return test_auc, test_acc, cm, recall, precision


if args.model != 'Transformer':
    args.repeat = 60

_, p_idx, labels_, cell_type, patient_id, data, cell_type_64 = Covid_data(args)
rkf = RepeatedKFold(n_splits=abs(args.n_splits), n_repeats=args.repeat*100, random_state=args.seed)
num = np.arange(len(p_idx))
accuracy, aucs, cms, recalls, precisions = [], [], [], [], []
iter_count = 0

for train_index, test_index in rkf.split(num):
    if args.n_splits < 0:
        temp_idx = train_index
        train_index = test_index
        test_index = temp_idx

    label_stat = []
    for idx in train_index:
        label_stat.append(labels_[p_idx[idx][0]])
    unique, cts = np.unique(label_stat, return_counts=True)
    if len(unique) < 2 or (1 in cts):
        continue
    print(dict(zip(unique, cts)))

    kk = 0
    while True:
        train_index_, valid_index, ty, vy = train_test_split(train_index, label_stat, test_size=0.33,
                                                              random_state=args.seed + kk)
        if len(set(ty)) == 2 and len(set(vy)) == 2:
            break
        kk += 1

    train_index = train_index_
    len_valid = len(valid_index)
    _index = np.concatenate([valid_index, test_index])

    train_ids = []
    for i in train_index:
        train_ids.append(patient_id[p_idx[i][0]])
    print(train_ids)

    x_train = []
    x_test = []
    x_valid = []
    y_train = []
    y_valid = []
    y_test = []
    id_train = []
    id_test = []
    id_valid = []
    # only use the augmented data (intra-mixup, inter-mixup) as train data
    data_augmented, train_p_idx, labels_augmented, cell_type_augmented = mixups(args, data, [p_idx[idx] for idx in train_index], labels_,
                                                           cell_type)
    individual_train, individual_test = sampling(args, train_p_idx, [p_idx[idx] for idx in _index], labels_,
                                                 labels_augmented, cell_type_augmented)
    for t in individual_train:
        id, label = [id_l[0] for id_l in t], [id_l[1] for id_l in t]
        x_train += [ii for ii in id]
        y_train += (label)
        id_train += (id)

    temp_idx = np.arange(len(_index))
    for t_idx in temp_idx[len_valid:]:
        id, label = [id_l[0] for id_l in individual_test[t_idx]], [id_l[1] for id_l in individual_test[t_idx]]
        x_test.append([ii for ii in id])
        y_test.append(label[0])
        id_test.append(id)
    for t_idx in temp_idx[:len_valid]:
        id, label = [id_l[0] for id_l in individual_test[t_idx]], [id_l[1] for id_l in individual_test[t_idx]]
        x_valid.append([ii for ii in id])
        y_valid.append(label[0])
        id_valid.append(id)
    x_train, x_valid, x_test, y_train, y_valid, y_test = x_train, x_valid, x_test, np.array(y_train).reshape([-1, 1]), \
                                                         np.array(y_valid).reshape([-1, 1]), np.array(y_test).reshape(
        [-1, 1])
    auc, acc, cm, recall, precision = train(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, data_augmented, data)
    aucs.append(auc)
    accuracy.append(acc)
    cms.append(cm)
    recalls.append(recall)
    precisions.append(precision)
    iter_count += 1
    if iter_count == abs(args.n_splits) * args.repeat:
        break

    del data_augmented

print("Best performance: Test ACC %f,   Test AUC %f,   Test Recall %f,   Test Precision %f" % (np.average(accuracy), np.average(aucs), np.average(recalls), np.average(precisions)))
accuracy = np.array(accuracy).reshape([-1, args.repeat]).mean(0)
aucs = np.array(aucs).reshape([-1, args.repeat]).mean(0)
recalls = np.array(recalls).reshape([-1, args.repeat]).mean(0)
precisions = np.array(precisions).reshape([-1, args.repeat]).mean(0)
ci_1 = st.t.interval(alpha=0.95, df=len(accuracy)-1, loc=np.mean(accuracy), scale=st.sem(accuracy))[1] - np.mean(accuracy)
ci_2 = st.t.interval(alpha=0.95, df=len(aucs)-1, loc=np.mean(aucs), scale=st.sem(aucs))[1] - np.mean(aucs)
ci_3 = st.t.interval(alpha=0.95, df=len(recalls)-1, loc=np.mean(recalls), scale=st.sem(recalls))[1] - np.mean(recalls)
ci_4 = st.t.interval(alpha=0.95, df=len(precisions)-1, loc=np.mean(precisions), scale=st.sem(precisions))[1] - np.mean(precisions)
print("ci: ACC ci %f,   AUC ci %f,   Recall ci %f,   Precision ci %f" % (ci_1, ci_2, ci_3, ci_4))
print(np.average(cms, 0))
print(patient_summary)
print(stats)

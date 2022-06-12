#!/usr/bin/env python

import argparse
import argcomplete

parser = argparse.ArgumentParser("Plotting clusters.")
parser.add_argument("figroot", nargs="+", type=str, default=["fig/lupus_1_"])

argcomplete.autocomplete(parser)
args = parser.parse_args()

import code
import scipy.sparse
import scipy
import copy
import math
import torch
import sys
sys.path.append(".")
import cloudpred
import datetime
import os
import logging.config
import traceback
import random
import numpy as np
import time
import pathlib
import seaborn as sns
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(args=None):

    model = []
    Xvalid = []
    Xtest = []
    for f in args.figroot:
        m = torch.load(f + "model.pt")
        m.eval()
        model.append(m)

        with open(args.figroot[0] + "Xvalid.pkl", "rb") as f:
            Xvalid.append(pickle.load(f))
        with open(args.figroot[0] + "Xtest.pkl", "rb") as f:
            Xtest.append(pickle.load(f))

    cloudpred.utils.latexify()

    # Compute t-SNE once for consistency
    samples = 5000
    import sklearn.manifold
    xvis = np.array(np.concatenate([i[0] for i in Xtest[0]]))
    perm = np.random.permutation(xvis.shape[0])
    tsne = sklearn.manifold.TSNE(n_components=2)
    tsne = tsne.fit_transform(xvis[perm[:samples], :])
    # tsne, mu = tsne[:perm.shape[0], :], tsne[perm.shape[0]:, :]

    for index in range(len(model)):
        x = np.array(np.concatenate([i[0][:, :2] for i in Xtest[index]]))
        c = np.concatenate([i[2] for i in Xtest[index]])
        ct = np.unique(c)

        ind = -np.ones(c.shape, np.int)
        for (i, t) in enumerate(ct):
            ind[c == t] = i
        color = sns.color_palette("hls", ct.size)
        handle = [matplotlib.patches.Patch(color=color[i], label=ct[i]) for i in range(ct.size)]
        color = np.array([list(color[i]) + [1] for i in ind])

        params = copy.deepcopy(model[index].pl.state_dict())
        ind = None
        best = -float("inf")
        auc = []
        res = []
        for c in range(model[index].pl.polynomial[0].centers):
            model[index].pl.polynomial[0].a.data[:, :c] = 0
            model[index].pl.polynomial[0].a.data[:, (c + 1):] = 0
            res.append(cloudpred.cloudpred.eval(model[index], Xtest[index]))
            print(res[-1], flush=True)
            if res[-1]["auc"] > best:
                ind = c
                best = res[-1]["auc"]
            model[index].pl.load_state_dict(params)
            auc.append(res[-1]["auc"])

        print("        Single Cluster Cross-entropy: " + str(res[ind]["ce"]))
        print("        Single Cluster Accuracy:      " + str(res[ind]["accuracy"]))
        print("        Single Cluster Soft Accuracy: " + str(res[ind]["soft"]))
        print("        Single Cluster AUC:           " + str(res[ind]["auc"]))
        print("        Single Cluster Coefficients:  " + str(model[index].pl.polynomial[0].a[:, ind]))

        x = torch.Tensor(np.array(np.concatenate([i[0] for i in Xtest[index]])))
        logp = torch.cat([c(x).unsqueeze(0) for c in model[index].mixture.component])
        shift, _ = torch.max(logp, 0)
        p = torch.exp(logp - shift) * model[index].mixture.weights
        p /= torch.sum(p, 0)
        c = np.concatenate([i[2] for i in Xtest[index]])

        for i in ct:
            print("Percent of {} Assigned to Best Cluster: {}".format(i, p[:, np.arange(c.shape[0])[c == i]].mean(1)[ind]))
        total = torch.sum(p[ind, :])
        for i in ct:
            ct_total = torch.sum(p[ind, np.arange(c.shape[0])[c == i]])
            print("Percent Best Cluster Composed of {}: {}".format(i, ct_total / total))


        fig = plt.figure(figsize=(2, 2))
        ax = plt.gca()
        plt.scatter(tsne[:, 0], tsne[:, 1], c=color[perm[:samples]], marker=".", s=1, linewidth=0, edgecolors="none", rasterized=True)
        plt.xticks([])
        plt.yticks([])

        logp = torch.cat([c(torch.as_tensor(xvis[perm[:samples], :], dtype=torch.float32)).unsqueeze(0) for c in model[index].mixture.component])
        shift, _ = torch.max(logp, 0)
        p = torch.exp(logp - shift) * model[index].mixture.weights
        p /= torch.sum(p, 0)

        xmin, xmax, ymin, ymax = plt.axis()

        for (i, m) in enumerate(model[index].mixture.component):
            if i == ind:
                c = "k"
                zorder = 2
                linewidth=1
            else:
                c = "gray"
                zorder = 1
                linewidth=0.5

            X = tsne[:, 0]
            Y = tsne[:, 1]
            Z = p[i, :].detach().numpy()

            closest = np.argmin(np.sum((xvis[perm[:samples], :] - m.mu.detach().numpy()) ** 2, 1))

            e = matplotlib.patches.Ellipse(tsne[closest, :], 0.10 * (xmax - xmin), 0.10 * (ymax - ymin),
                     angle=0, linewidth=linewidth, fill=False, zorder=zorder, edgecolor=c)

            ax.add_patch(e)

        pathlib.Path(os.path.dirname(args.figroot[index])).mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.figroot[index] + "tsne.pdf", dpi=600)


        fig = plt.figure(figsize=(2, 2))
        ax = plt.gca()

        plt.scatter(x[perm, 0], x[perm, 1], c=color[perm], marker=".", s=1, linewidth=0, edgecolors="none", rasterized=True)
        plt.xticks([])
        plt.yticks([])

        for (i, m) in enumerate(model[index].mixture.component):
            if i == ind:
                color = "k"
                zorder = 2
                linewidth=1
            else:
                color = "gray"
                zorder = 1
                linewidth=0.5
            e = matplotlib.patches.Ellipse(m.mu[:2], 3 / math.sqrt(max(abs(m.invvar[0]), 1e-5)), 3 / math.sqrt(max(abs(m.invvar[1]), 1e-5)),
             angle=0, linewidth=linewidth, fill=False, zorder=zorder, edgecolor=color)
            ax.add_patch(e)

        pathlib.Path(os.path.dirname(args.figroot[index])).mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.figroot[index] + "clusters.pdf", dpi=600)

        fig = plt.figure(figsize=(2, 3))
        plt.legend(handles=handle, loc="center", fontsize="small")
        plt.title("")
        plt.axis("off")
        plt.savefig(args.figroot[index] + "legend.pdf")

if __name__ == "__main__":
    main(args)

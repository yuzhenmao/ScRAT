import cloudpred
import numpy as np
import sklearn
import torch
import math
import torch


def train(Xtrain, Xvalid, centers=2, regression=False):
    outputs = 1 if regression else 2
    classifier = torch.nn.Sequential(torch.nn.Linear(Xtrain[0][0].shape[1], centers), torch.nn.ReLU(), torch.nn.Linear(centers, centers), torch.nn.ReLU(), cloudpred.utils.Aggregator(), torch.nn.Linear(centers, centers), torch.nn.ReLU(), torch.nn.Linear(centers, outputs))
    reg = None
    return cloudpred.utils.train_classifier(Xtrain, Xvalid, [], classifier, regularize=reg, iterations=1000, eta=1e-4, stochastic=True, regression=regression)


def eval(model, Xtest, regression=False):
    reg = None
    model, res =  cloudpred.utils.train_classifier([], Xtest, [], model, regularize=reg, iterations=1, eta=0, stochastic=True, regression=regression)
    return res

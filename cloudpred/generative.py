import collections
import numpy as np
import sklearn.mixture
import math
import logging


def train(Xtrain, centers=2):
    gm = collections.defaultdict(list)
    count = collections.defaultdict(int)
    for X, y, *_ in Xtrain:
        gm[y].append(X)
        count[y] += 1
    
    for state in gm:
        gm[state] = np.concatenate(gm[state])
        model = sklearn.mixture.GaussianMixture(centers)
        gm[state] = model.fit(gm[state])

    return (gm, count)

def eval(model, Xtest):
    gm, count = model
    total = 0.
    correct = 0
    prob = 0.
    y_score = []
    y_true = []
    for X, y, *_ in Xtest:
        logp = {}
        x = -float("inf")
        for state in gm:
            logp[state] = sum(gm[state].score_samples(X))
            x = max(x, logp[state])
        y_score.append(logp[1] - logp[0])
        y_true.append(y)
        Z = 0
        for state in logp:
            logp[state] = math.exp(logp[state] - x) * count[state]
            Z += logp[state]
        pred = None
        for state in logp:
            logp[state] /= Z
            if pred is None or logp[state] > logp[pred]:
                pred = state
        # total += math.log(logp[state])
        correct += (pred == y)
        prob += logp[y]
    n = len(Xtest)

    res = {}
    res["accuracy"] = correct / float(n)
    res["soft"] = prob / float(n)
    res["auc"] = sklearn.metrics.roc_auc_score(y_true, y_score)

    logger = logging.getLogger(__name__)
    # logger.debug("        Generative Cross-entropy: " + str(total / float(n)))
    logger.debug("        Generative Accuracy:      " + str(res["accuracy"]))
    logger.debug("        Generative Soft Accuracy: " + str(res["soft"]))
    logger.debug("        Generative AUC:           " + str(res["auc"]))

    return res

import collections
import sklearn
import sys
import math
import logging


def train(Xtrain, centers=2):
    gm = collections.defaultdict(list)
    for (i, (X, y, *_)) in enumerate(Xtrain):
        model = sklearn.mixture.GaussianMixture(min(centers, X.shape[0]))
        model.fit(X)
        gm[y].append(model)
    
        print("Train " + str(i + 1) + " / " + str(len(Xtrain)), end="\r")
        sys.stdout.flush()
    print()
    return gm

def eval(gm, Xtest):
    total = 0.
    correct = 0
    prob = 0.
    y_score = []
    y_true = []
    for (i, (X, y, *_)) in enumerate(Xtest):
        logp = {}
        x = -float("inf")
        for state in gm:
            logp[state] = list(map(lambda m: sum(m.score_samples(X)), gm[state]))
            x = max(x, max(logp[state]))
        # print(logp)
        Z = 0
        for state in logp:
            logp[state] = sum(map(lambda lp: math.exp(lp - x), logp[state]))
            Z += logp[state]
        pred = None
        for state in logp:
            logp[state] /= Z
            if pred is None or logp[state] > logp[pred]:
                pred = state
        total += -math.log(max(1e-50, logp[y]))
        correct += (pred == y)
        prob += logp[y]
        y_score.append(logp[1])
        y_true.append(y)
    
        print("Test " + str(i + 1) + " / " + str(len(Xtest)) + ": " + str(correct / float(i + 1)), end="\r", flush=True)
    print()
    
    n = len(Xtest)
    res = {}
    res["ce"] = total / float(n)
    res["accuracy"] = correct / float(n)
    res["soft"] = prob / float(n)
    res["auc"] = sklearn.metrics.roc_auc_score(y_true, y_score)

    logger = logging.getLogger(__name__)
    logger.debug("        Genpat Cross-entropy: " + str(res["ce"]))
    logger.debug("        Genpat Accuracy:      " + str(res["accuracy"]))
    logger.debug("        Genpat Soft Accuracy: " + str(res["soft"]))
    logger.debug("        Genpat AUC:           " + str(res["auc"]))

    return res

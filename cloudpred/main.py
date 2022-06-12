import copy
import math
import scipy
import torch
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
import pdb
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(args=None):

    # Parse command line arguments and set up logging
    parser = cloudpred.parser()
    args = parser.parse_args()
    cloudpred.utils.setup_logging(args.logfile, args.loglevel)
    logger = logging.getLogger(__name__)
    logger.info(args)

    try:
        # Seeding RNGs
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        t = time.time()
        Xtrain, Xvalid, Xtest, state = cloudpred.utils.load_synthetic(args.dir, valid=args.valid, test=args.test, train_patients=args.train_patients, cells=args.cells)
        logger.debug("Loading data took " + str(time.time() - t))

        ### Transform data ###
        t = time.time()
        if args.transform == "none":
            pass
        elif args.transform == "log":
            Xtrain = list(map(lambda x: (x[0].log1p(), *x[1:]), Xtrain))
            Xvalid = list(map(lambda x: (x[0].log1p(), *x[1:]), Xvalid))
            Xtest  = list(map(lambda x: (x[0].log1p(), *x[1:]), Xtest))
        else:
            message = "Transform " + args.transform + " is not implemented."
            raise NotImplementedError(message)

        if args.pc:
            dims = 50
            iterations = 5
            try:
                pc = np.load(args.dir + "/pc_" + args.transform + "_" + str(args.seed) + "_" + str(args.dims) + "_" + str(iterations) + ".npz")["pc"]
            except FileNotFoundError:
                pc = cloudpred.utils.train_pca_autoencoder(scipy.sparse.vstack(map(lambda x: x[0], Xtrain)), None,
                                                           scipy.sparse.vstack(map(lambda x: x[0], Xvalid)), None,
                                                           args.dims, args.transform,
                                                           iterations=iterations,
                                                           figroot=args.figroot) # TODO: get rid of figroot?
                np.savez_compressed(args.dir + "/pc_" + args.transform + "_" + str(args.seed) + "_" + str(args.dims) + "_" + str(iterations) + ".npz",
                                    pc=pc)

            pc = pc[:, :args.dims]

            ### Project onto principal components ###
            mu = scipy.sparse.vstack(list(map(lambda x: x[0], Xtrain))).mean(axis=0)
            Xtrain = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xtrain))  # - np.asarray(mu.dot(pc))
            Xvalid = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xvalid))  # - np.asarray(mu.dot(pc))
            Xtest  = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xtest))   # - np.asarray(mu.dot(pc))
            full = np.concatenate(list(map(lambda x: x[0], Xtrain)))
            mu = np.mean(full, axis=0)
            sigma = np.sqrt(np.mean(np.square(full - mu), axis=0))
            sigma = sigma[0, 0]
            Xtrain = list(map(lambda x: ((x[0] - mu) / sigma, *x[1:]), Xtrain))  # - np.asarray(mu.dot(pc))
            Xvalid = list(map(lambda x: ((x[0] - mu) / sigma, *x[1:]), Xvalid))  # - np.asarray(mu.dot(pc))
            Xtest  = list(map(lambda x: ((x[0] - mu) / sigma, *x[1:]), Xtest))   # - np.asarray(mu.dot(pc))
        else:
            Xtrain = list(map(lambda x: (x[0].todense(), *x[1:]), Xtrain))
            Xvalid = list(map(lambda x: (x[0].todense(), *x[1:]), Xvalid))
            Xtest  = list(map(lambda x: (x[0].todense(), *x[1:]), Xtest))
        logger.debug("Transforming data took " + str(time.time() - t))

        pdb.set_trace()


        ### Train model ###
        if args.cloudpred:
            best_model = None
            best_score = float("inf")
            for centers in args.centers:
                model, res = cloudpred.cloudpred.train(Xtrain, Xvalid, centers, regression=args.regression)
                if res["loss"] < best_score:
                    best_model = model
                    best_score = res["loss"]
                    best_centers = centers

            if args.figroot is not None:
                pathlib.Path(os.path.dirname(args.figroot)).mkdir(parents=True, exist_ok=True)
                torch.save(best_model, args.figroot + "model.pt")
                with open(args.figroot + "Xvalid.pkl", "wb") as f:
                    pickle.dump(Xvalid, f)
                with open(args.figroot + "Xtest.pkl", "wb") as f:
                    pickle.dump(Xtest, f)
                print(best_model)
                print(best_model.pl)

                x = np.array(np.concatenate([i[0][:, :2] for i in Xtest]))
                c = np.concatenate([i[2] for i in Xtest])
                ct = np.unique(c)
                print(ct)
                print(c)
                print(c.shape)
                ind = -np.ones(c.shape, np.int)
                for (i, t) in enumerate(ct):
                    ind[c == t] = i
                print(ind)
                color = sns.color_palette("hls", ct.size)
                handle = [matplotlib.patches.Patch(color=color[i], label=ct[i]) for i in range(ct.size)]
                color = np.array([list(color[i]) + [1] for i in ind])

                params = copy.deepcopy(best_model.pl.state_dict())
                ind = None
                best = -float("inf")
                auc = []
                res = []
                criterion = "r2" if args.regression else "auc"
                for c in range(best_model.pl.polynomial[0].centers):
                    best_model.pl.polynomial[0].a.data[:, :c] = 0
                    best_model.pl.polynomial[0].a.data[:, (c + 1):] = 0
                    print(best_model.pl.polynomial[0].a)
                    res.append(cloudpred.cloudpred.eval(best_model, Xtest, regression=args.regression))
                    print(res[-1], flush=True)
                    if res[-1][criterion] > best:
                        ind = c
                        best = res[-1][criterion]
                    best_model.pl.load_state_dict(params)  # TODO: needs to be here for final eval
                    auc.append(res[-1][criterion])

                logger.info("        Single Cluster Loss:          " + str(res[ind]["loss"]))
                logger.info("        Single Cluster Accuracy:      " + str(res[ind]["accuracy"]))
                logger.info("        Single Cluster Soft Accuracy: " + str(res[ind]["soft"]))
                logger.info("        Single Cluster AUC:           " + str(res[ind]["auc"]))
                logger.info("        Single Cluster R2:            " + str(res[ind]["r2"]))
                logger.info("        Single Cluster Coefficients:  " + str(best_model.pl.polynomial[0].a[:, ind]))

                x = torch.Tensor(np.array(np.concatenate([i[0] for i in Xtest])))
                logp = torch.cat([c(x).unsqueeze(0) for c in best_model.mixture.component])
                shift, _ = torch.max(logp, 0)
                p = torch.exp(logp - shift) * best_model.mixture.weights
                p /= torch.sum(p, 0)
                c = np.concatenate([i[2] for i in Xtest])

                for i in ct:
                    logger.info("Percent of {} Assigned to Best Cluster: {}".format(i, p[:, np.arange(c.shape[0])[c == i]].mean(1)[ind]))
                total = torch.sum(p[ind, :])
                for i in ct:
                    ct_total = torch.sum(p[ind, np.arange(c.shape[0])[c == i]])
                    logger.info("Percent Best Cluster Composed of {}: {}".format(i, ct_total / total))

                cloudpred.utils.latexify()

                import sklearn.manifold
                fig = plt.figure(figsize=(2, 2))
                ax = plt.gca()
                print(x.shape)
                perm = np.random.permutation(x.shape[0])[:5000]
                print(x[perm, :].shape)
                print([m.mu.detach().numpy().shape for m in model.mixture.component])
                tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(np.concatenate([x[perm, :]] + [np.expand_dims(m.mu.detach().numpy(), 0) for m in model.mixture.component]))
                tsne, mu = tsne[:perm.shape[0], :], tsne[perm.shape[0]:, :]
                print(tsne.shape)
                print(perm.shape)
                print(mu.shape)
                print(x.shape)
                print(color.shape)
                plt.scatter(tsne[:, 0], tsne[:, 1], c=color[perm, :], marker=".", s=1, linewidth=0, edgecolors="none", rasterized=True)
                plt.xticks([])
                plt.yticks([])

                xmin, xmax, ymin, ymax = plt.axis()
                for (i, m) in enumerate(mu):
                    if i == ind:
                        c = "k"
                        zorder = 2
                        linewidth=1
                    else:
                        c = "gray"
                        zorder = 1
                        linewidth=0.5
                    e = matplotlib.patches.Ellipse(m, 0.10 * (xmax - xmin), 0.10 * (ymax - ymin),
                     angle=0, linewidth=linewidth, fill=False, zorder=zorder, edgecolor=c)
                    ax.add_patch(e)

                pathlib.Path(os.path.dirname(args.figroot)).mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(args.figroot + "tsne.pdf", dpi=600)


                fig = plt.figure(figsize=(2, 2))
                ax = plt.gca()


                perm = np.random.permutation(x.shape[0])
                print(perm)
                plt.scatter(x[perm, 0], x[perm, 1], c=color[perm], marker=".", s=1, linewidth=0, edgecolors="none", rasterized=True)
                plt.xticks([])
                plt.yticks([])

                for (i, m) in enumerate(model.mixture.component):
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

                pathlib.Path(os.path.dirname(args.figroot)).mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(args.figroot + "clusters.pdf", dpi=600)

                fig = plt.figure(figsize=(2, 2))
                plt.legend(handles=handle, loc="center", fontsize="xx-small")
                plt.title("")
                plt.axis("off")
                # plt.tight_layout()
                plt.savefig(args.figroot + "legend.pdf")

            res = cloudpred.cloudpred.eval(best_model, Xtest, regression=args.regression)
            logger.info("        CloudPred Loss:          " + str(res["loss"]))
            logger.info("        CloudPred Accuracy:      " + str(res["accuracy"]))
            logger.info("        CloudPred Soft Accuracy: " + str(res["soft"]))
            logger.info("        CloudPred AUC:           " + str(res["auc"]))
            logger.info("        CloudPred R2:            " + str(res["r2"]))
            logger.info("        CloudPred Centers:       " + str(best_centers))


        ### Basic classifier ###
        if args.linear:
            linear = torch.nn.Sequential(cloudpred.utils.Aggregator(), Linear(Xtrain[0][0].shape[1], len(state)))
            model, res = cloudpred.utils.train_classifier(Xtrain, [], [], linear, eta=1e-3, iterations=100, state=state, regression=args.regression)
            model, res =  cloudpred.utils.train_classifier([], Xtest, [], model, regularize=None, iterations=1, eta=0, stochastic=True, regression=args.regression)
            logger.info("        Linear Loss:          " + str(res["loss"]))
            logger.info("        Linear Accuracy:      " + str(res["accuracy"]))
            logger.info("        Linear Soft Accuracy: " + str(res["soft"]))
            logger.info("        Linear AUC:           " + str(res["auc"]))
            logger.info("        Linear R2:            " + str(res["r2"]))


        ### Generative models ###
        if args.generative:
            best_model = None
            best_score = -float("inf")
            for centers in args.centers:
                model = cloudpred.generative.train(Xtrain, centers)
                logger.debug("    Training:")
                res = cloudpred.generative.eval(model, Xtrain)
                logger.debug("    Validation")
                res = cloudpred.generative.eval(model, Xvalid)
                if res["accuracy"] > best_score:
                    best_model = model
                    best_score = res["accuracy"]
            logger.debug("    Testing:")
            res = cloudpred.generative.eval(best_model, Xtest)
            logger.info("        Generative Accuracy:      " + str(res["accuracy"]))
            logger.info("        Generative Soft Accuracy: " + str(res["soft"]))
            logger.info("        Generative AUC:           " + str(res["auc"]))

        if args.genpat:
            best_model = None
            best_score = -float("inf")
            for centers in args.centers:
                model = cloudpred.genpat.train(Xtrain, centers)
                logger.debug("    Training:")
                res = cloudpred.genpat.eval(model, Xtrain)
                logger.debug("    Validation:")
                res = cloudpred.genpat.eval(model, Xvalid)
                if res["accuracy"] > best_score:
                    best_model = model
                    best_score = res["accuracy"]
            logger.debug("    Testing:")
            res = cloudpred.genpat.eval(best_model, Xtest)
            logger.info("        Genpat Loss:          " + str(res["ce"]))
            logger.info("        Genpat Accuracy:      " + str(res["accuracy"]))
            logger.info("        Genpat Soft Accuracy: " + str(res["soft"]))
            logger.info("        Genpat AUC:           " + str(res["auc"]))


        if args.deepset:
            best_model = None
            best_score = -float("inf")
            for centers in args.centers:
                model, res = cloudpred.deepset.train(Xtrain, Xvalid, centers, regression=args.regression)
                if res["accuracy"] > best_score:
                    best_model = model
                    best_score = res["accuracy"]
            res = cloudpred.deepset.eval(best_model, Xtest, regression=args.regression)
            logger.info("        DeepSet Loss:          " + str(res["loss"]))
            logger.info("        DeepSet Accuracy:      " + str(res["accuracy"]))
            logger.info("        DeepSet Soft Accuracy: " + str(res["soft"]))
            logger.info("        DeepSet AUC:           " + str(res["auc"]))
            logger.info("        DeepSet R2:            " + str(res["r2"]))

    except Exception as e:
        logger.exception(traceback.format_exc())
        raise


class Linear(torch.nn.Module):
    def __init__(self, dim, states):
         super(Linear, self).__init__()
         self.states = states
         if states == 1:
             self.layer = torch.nn.Linear(dim, states)
         else:
             self.layer = torch.nn.Linear(dim, states - 1)
         self.layer.weight.data.zero_()
         if self.layer.bias is not None:
             self.layer.bias.data.zero_()

    def forward(self, x):
        if self.states == 1:
            return torch.sum(self.layer(x), dim=-2, keepdim=True)
        else:
            return torch.cat([torch.zeros(1, 1), torch.sum(self.layer(x), dim=-2, keepdim=True)], dim=1)

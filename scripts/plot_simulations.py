#!/usr/bin/env python

"""Plot results for simulations."""

import math
import os
import pathlib

import argparse
import argcomplete
import bootstrapped.bootstrap
import bootstrapped.stats_functions
import cloudpred
import matplotlib.pyplot as plt
import numpy as np
import tqdm

parser = argparse.ArgumentParser("Simulation plots.")
parser.add_argument("dir", nargs="?", type=str, default="log")

argcomplete.autocomplete(parser)
args = parser.parse_args()


def load(filename):
    cloudpred = float("nan")
    linear = float("nan")
    generative = float("nan")
    genpat = float("nan")
    deepset = float("nan")

    try:
        cache_name = filename + "_cache"
        if os.path.isfile(cache_name) and os.path.getctime(cache_name) > os.path.getctime(filename):
            with open(cache_name, "r") as f:
                lines = [line for line in f if line[:4] == "INFO"]
        else:
            with open(filename, "r") as f:
                lines = [line for line in f if line[:4] == "INFO"]
            with open(cache_name, "w") as f:
                for l in lines:
                    f.write(l)

        for line in lines:
            if "CloudPred AUC:" in line:
                cloudpred = float(line.split()[-1])
            if "Linear AUC:" in line:
                linear = float(line.split()[-1])
            if "Generative AUC:" in line:
                generative = float(line.split()[-1])
            if "Genpat AUC:" in line:
                genpat = float(line.split()[-1])
            if "DeepSet AUC:" in line:
                deepset = float(line.split()[-1])
    except FileNotFoundError:
        pass
    return cloudpred, linear, generative, genpat, deepset


def load_all(root, seeds):
    x = []
    for seed in tqdm.tqdm(seeds):
        x.append(load(root + seed))
    return list(zip(*x))


if __name__ == "__main__":
    seeds = 25
    dim = 25
    cloudpred.utils.latexify()
    pathlib.Path("fig").mkdir(parents=True, exist_ok=True)

    ### Legend ###
    variation = "1.0"
    splits = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    n = 200

    plt.figure(figsize=(1.79, 1.79))
    for i in range(5):
        plt.plot([float("nan"), float("nan")])
    plt.title("")
    plt.axis("off")
    plt.legend(["CloudPred", "Independent", "Generative (class)", "Generative (patient)", "DeepSet"], loc="center", fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join("fig", "legend.pdf"))

    ### Varying signature ###
    variation = "1.0"
    splits = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    n = 200

    x = []
    for split in splits:
        x.append(load_all(os.path.join(args.dir, "signature_{}_{}_{}_{}_".format(dim, variation, split, n)), map(str, range(1, seeds + 1))))
    print(x)

    splits = list(map(float, splits))
    plt.figure(figsize=(1.79, 1.79))
    for x in zip(*x):
        xerr = list(map(lambda t: bootstrapped.bootstrap.bootstrap(np.array([u for u in t if not math.isnan(u)]), stat_func=bootstrapped.stats_functions.mean), x))
        xerr = list(map(lambda t: (t.lower_bound, t.upper_bound), xerr))
        xerr = np.array(xerr).transpose()
        x = list(map(np.nanmean, x))
        xerr -= x
        xerr = abs(xerr)
        plt.errorbar(splits, x, xerr)
    plt.title("")
    plt.xlabel("Signature distribution")
    plt.ylabel("AUC")
    plt.axis([-0.03, 1.03, 0.5, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join("fig", "split_{}.pdf".format(dim)))

    ### Training set size (mixed signature) ###
    variation = "1.0"
    split = "0.5"
    examples = [110, 120, 130, 140, 150, 200, 250, 300]
    examples = [110, 150, 200, 250]

    x = []
    for n in examples:
        x.append(load_all(os.path.join(args.dir, "signature_{}_{}_{}_{}_".format(dim, variation, split, n)), map(str, range(1, seeds + 1))))

    examples = list(map(lambda x: 2 * x - 200, examples))

    plt.figure(figsize=(1.79, 1.79))
    for x in zip(*x):
        print(x)
        xerr = list(map(lambda t: bootstrapped.bootstrap.bootstrap(np.array([u for u in t if not math.isnan(u)]), stat_func=bootstrapped.stats_functions.mean), x))
        xerr = list(map(lambda t: (t.lower_bound, t.upper_bound), xerr))
        xerr = np.array(xerr).transpose()
        x = list(map(np.nanmean, x))
        xerr -= x
        xerr = abs(xerr)
        print(xerr)
        print(x)
        plt.errorbar(examples, x, xerr)
    plt.title("")
    plt.xlabel("Training set size")
    plt.ylabel("AUC")
    plt.axis([0.0, 312, 0.5, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join("fig", "examples_{}.pdf".format(dim)))

    ### Variation (mixed signature)
    variations = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    split = "0.5"
    n = 200

    x = []
    for variation in variations:
        x.append(load_all(os.path.join(args.dir, "signature_{}_{}_{}_{}_".format(dim, variation, split, n)), map(str, range(1, seeds + 1))))

    variations = list(map(float, variations))
    plt.figure(figsize=(1.79, 1.79))
    for x in zip(*x):
        xerr = list(map(lambda t: bootstrapped.bootstrap.bootstrap(np.array([u for u in t if not math.isnan(u)]), stat_func=bootstrapped.stats_functions.mean), x))
        xerr = list(map(lambda t: (t.lower_bound, t.upper_bound), xerr))
        xerr = np.array(xerr).transpose()
        x = list(map(np.nanmean, x))
        xerr -= x
        xerr = abs(xerr)
        plt.errorbar(variations, x, xerr)
    plt.title("")
    plt.xlabel("Variation")
    plt.ylabel("AUC")
    plt.axis([-0.03, 1.03, 0.5, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join("fig", "variation_{}.pdf".format(dim)))

    ### Training Patients (interaction) ###
    variation = "1.0"
    examples = [110, 120, 130, 140, 150, 200, 250, 300]
    # examples = [110, 150, 200, 250, 300]
    examples = [110, 150, 200, 250]

    x = []
    for n in examples:
        x.append(load_all(os.path.join(args.dir, "interaction_{}_{}_{}_".format(dim, variation, n)), map(str, range(1, seeds + 1))))

    examples = list(map(lambda x: 2 * x - 200, examples))

    plt.figure(figsize=(1.79, 1.79))
    for x in zip(*x):
        xerr = list(map(lambda t: bootstrapped.bootstrap.bootstrap(np.array([u for u in t if not math.isnan(u)]), stat_func=bootstrapped.stats_functions.mean), x))
        xerr = list(map(lambda t: (t.lower_bound, t.upper_bound), xerr))
        xerr = np.array(xerr).transpose()
        x = list(map(np.nanmean, x))
        xerr -= x
        xerr = abs(xerr)
        plt.errorbar(examples, x, xerr)
    plt.title("")
    plt.xlabel("Training set size")
    plt.ylabel("AUC")
    plt.axis([0.0, 312, 0.5, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join("fig", "examples_interaction_{}.pdf".format(dim)))

    ### Variation (interaction) ###
    variations = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    n = 200

    x = []
    for variation in variations:
        x.append(load_all(os.path.join(args.dir, "interaction_{}_{}_{}_".format(dim, variation, n)), map(str, range(1, seeds + 1))))

    variations = list(map(float, variations))
    plt.figure(figsize=(1.79, 1.79))
    for x in zip(*x):
        xerr = list(map(lambda t: bootstrapped.bootstrap.bootstrap(np.array([u for u in t if not math.isnan(u)]), stat_func=bootstrapped.stats_functions.mean), x))
        xerr = list(map(lambda t: (t.lower_bound, t.upper_bound), xerr))
        xerr = np.array(xerr).transpose()
        x = list(map(np.nanmean, x))
        xerr -= x
        xerr = abs(xerr)
        plt.errorbar(variations, x, xerr)
    plt.title("")
    plt.xlabel("Variation")
    plt.ylabel("AUC")
    plt.axis([-0.03, 1.03, 0.5, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join("fig", "variation_interaction_{}.pdf".format(dim)))

#!/usr/bin/env python3

"""Plot results for lupus experiments."""

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

parser = argparse.ArgumentParser("Lupus plots.")
parser.add_argument("dir", nargs="?", type=str, default="log")

argcomplete.autocomplete(parser)
args = parser.parse_args()


def load(filename_format):
    x = 5 * [float("nan")]

    for (index, m) in enumerate(["CloudPred", "Linear", "Generative", "Genpat", "DeepSet"]):
        try:
            filename = filename_format.format(m.lower())
            cache_name = filename + "_cache"
            if (os.path.isfile(cache_name)
                    and os.path.getctime(cache_name) > os.path.getctime(filename)):
                with open(cache_name, "r") as f:
                    lines = [line for line in f if line[:4] == "INFO"]
            else:
                with open(filename, "r") as f:
                    lines = [line for line in f if line[:4] == "INFO"]
                with open(cache_name, "w") as f:
                    for l in lines:
                        f.write(l)

            for line in lines:
                if "{} AUC:".format(m) in line and math.isnan(x[index]):
                    x[index] = float(line.split()[-1])
                if "{} R2:".format(m) in line and math.isnan(x[index]):
                    x[index] = float(line.split()[-1])
        except FileNotFoundError:
            pass

    cloudpred, linear, generative, genpat, deepset = x

    return cloudpred, linear, generative, genpat, deepset


def load_all(root, seeds):
    x = []
    for seed in tqdm.tqdm(seeds):
        x.append(load(root + seed))
    return list(zip(*x))


if __name__ == "__main__":
    seeds = 25
    dim = 25
    tasks = ["lupus", "lupus_pop", "mono"]
    cloudpred.utils.latexify()
    pathlib.Path("fig").mkdir(parents=True, exist_ok=True)

    # Printing numbers for results table
    for task in tasks:
        print(task)
        x = load_all(os.path.join(args.dir, "{}_".format(task)), map(str, range(1, seeds + 1)))
        print(list(map(lambda t: seeds - np.isnan(t).sum(), x)))
        xerr = list(map(lambda t: bootstrapped.bootstrap.bootstrap(np.array([u for u in t if not math.isnan(u)]) if not all(map(math.isnan, t)) else np.array([float("nan")]), stat_func=bootstrapped.stats_functions.mean), x))
        print(list(map(lambda t: "{:.3f} ({:.3f}-{:.3f})".format(t.value, t.lower_bound, t.upper_bound), xerr)))

    for task in tasks:
        print(task)
        TRAIN = [10, 20, 30, 40, 50, 60, 70]
        SEED = list(range(1, seeds + 1))
        x = np.zeros((len(TRAIN), len(SEED), 5))
        for (i, train) in tqdm.tqdm(enumerate(TRAIN)):
            for (j, seed) in enumerate(SEED):
                x[i, j, :] = np.array(load(os.path.join(args.dir, task, "{{}}_{}_train_{}".format(seed, train))))

        plt.figure(figsize=(1.79, 1.79))
        for i in range(5):
            err = list(map(lambda t: bootstrapped.bootstrap.bootstrap(np.array([u for u in t if not math.isnan(u)]), stat_func=bootstrapped.stats_functions.mean), x[:, :, i]))
            err = list(map(lambda t: (t.lower_bound, t.upper_bound), err))
            print(err)
            err = np.array(err).transpose()
            y = list(map(np.nanmean, x[:, :, i]))
            print(y)
            print(list(map(lambda t: np.sum(~np.isnan(t)), x[:, :, i])))
            print()
            err -= y
            err = abs(err)
            plt.errorbar(TRAIN, y, err)
        plt.title("")
        plt.xlabel("Training Patients")
        plt.ylabel("AUC")
        # plt.axis([-0.03, 1.03, 0.5, 1.0])
        plt.tight_layout()
        plt.savefig(os.path.join("fig", "{}_train.pdf".format(task)))
        print()
        print()

        CELLS = [1, 10, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
        SEED = list(range(1, seeds + 1))
        x = np.zeros((len(CELLS), len(SEED), 5))
        for (i, cells) in tqdm.tqdm(enumerate(CELLS)):
            for (j, seed) in enumerate(SEED):
                x[i, j, :] = np.array(load(os.path.join(args.dir, task, "{{}}_{}_cells_{}".format(seed, cells))))

        plt.figure(figsize=(1.79, 1.79))
        for i in range(5):
            err = list(map(lambda t: bootstrapped.bootstrap.bootstrap(np.array([u for u in t if not math.isnan(u)]), stat_func=bootstrapped.stats_functions.mean), x[:, :, i]))
            err = list(map(lambda t: (t.lower_bound, t.upper_bound), err))
            print(x[:, :, i])
            print(err)
            err = np.array(err).transpose()
            y = list(map(np.nanmean, x[:, :, i]))
            print(y)
            print(list(map(lambda t: np.sum(~np.isnan(t)), x[:, :, i])))
            print()
            err -= y
            err = abs(err)
            plt.errorbar(CELLS, y, err)
        plt.title("")
        plt.xlabel("Cells")
        plt.ylabel("AUC")
        # plt.axis([-0.03, 1.03, 0.5, 1.0])
        plt.tight_layout()
        plt.savefig(os.path.join("fig", "{}_cells.pdf".format(task)))

#!/usr/bin/env python3

"""Creates datasets for simulations using purified data."""

import argparse
import os
import pathlib
import random

import numpy as np
import scipy.io
import scipy.sparse


POPULATION = [
    "b_cells",
    "cd4_t_helper",
    "naive_t",
    "cd14_monocytes",
    "cd56_nk",
    "memory_t",
    "regulatory_t",
    "cd34",
    "cytotoxic_t",
    "naive_cytotoxic"
]


def main():
    """Creates datasets for simulations using purified data."""

    parser = argparse.ArgumentParser(description="Generate simulations")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="seed for RNG")
    parser.add_argument("-v", "--variation", type=str, default="0.0",
                        help="variation parameter (float)")
    parser.add_argument("--split", type=str, default="0.0",
                        help="split parameter (float)")
    parser.add_argument("-n", "--examples", type=int, default=200,
                        help="number of examples")
    parser.add_argument("--interaction", action="store_true", default=False,
                        help="interaction between two types")
    args = parser.parse_args()

    # Load purified data
    counts = []
    for p in POPULATION:
        c = scipy.sparse.load_npz(os.path.join("data", "purified", "{}.npz".format(p))).transpose()
        mean = np.mean(c, axis=1)
        # Faster version than running
        # >> c -= mean
        # >> c = scipy.sparse.csr_matrix(np.maximum(c, 0))
        # https://stackoverflow.com/questions/39685168/scipy-sparse-matrix-subtract-row-mean-to-nonzero-elements
        d = scipy.sparse.diags(np.array(mean).squeeze(1), 0)
        ones = c.copy()
        ones.data = np.ones_like(ones.data)
        x = c - (d * ones)
        x.data[x.data < 0] = 0
        counts.append(c)

    # Generate splits of cells
    count_train = []
    count_test = []

    random.seed(args.seed)
    np.random.seed(args.seed)

    for c in counts:
        index = np.arange(np.shape(c)[0])
        np.random.shuffle(index)
        c = c[index, :]
        count_train.append(c[:np.shape(c)[0] // 2, :])
        count_test.append(c[np.shape(c)[0] // 2:, :])

    # Create example patients, and save each patient to a file
    if args.interaction:
        path = os.path.join("data", "interaction",
                            "{}_{}_{}".format(args.variation, args.examples, args.seed))
        create = lambda disease, c: create_interaction(disease, c, float(args.variation))
        generate(path,
                 create,
                 count_train,
                 count_test,
                 args.examples)
    else:
        path = os.path.join("data", "signature",
                            "{}_{}_{}_{}".format(args.variation, args.split,
                                                 args.examples, args.seed))
        create = lambda disease, c: create_signature(disease, c,
                                                     float(args.variation), float(args.split))
        generate(path,
                 create,
                 count_train,
                 count_test,
                 args.examples)


def generate(root, create_example, count_train, count_test, n_examples=100):
    """Creates simulated patients."""

    pathlib.Path(root).mkdir(parents=True, exist_ok=True)

    for d in ["Disease", "Healthy", "Test"]:
        pathlib.Path(os.path.join(root, d)).mkdir(parents=True, exist_ok=True)

        for i in range(n_examples):
            if d == "Disease":
                disease = True
            elif d == "Healthy":
                disease = False
            elif d == "Test":
                disease = random.random() < 0.5
            else:
                raise NotImplementedError()

            if d == "Test":
                c = count_test
            else:
                c = count_train

            example = create_example(disease, c)
            if isinstance(example, np.ndarray):
                example = scipy.sparse.csr_matrix(example)
            counts = scipy.sparse.save_npz(
                os.path.join(root, d, "{}_{}".format(d.lower(), i)), example)

def create_signature(disease, c, variation, split):
    """Creates patient with a mixed signature."""
    cells = 1000

    def r(low, high):
        return variation * random.uniform(low, high) + (1 - variation) * (high + low) / 2.

    p = np.zeros(len(POPULATION))
    p[0] = r(4, 69)  # bcell
    p[1] = r(0, 49)  # mono
    p[2] = r(0, 50)
    p[3] = r(22, 93) # cd4
    p[4] = r(0, 59)  # nk cells
    p[5] = r(0, 50)
    p[6] = r(0, 50)
    p[7] = r(6, 65)  # CD8 cytotoxic
    p[8] = r(0, 50)
    p[9] = r(0, 50)

    if disease:
        if random.random() < split:
            p[3] = random.uniform(0, 22)
        else:
            p[3] = random.uniform(93, 115)

    p /= sum(p)

    pop = np.random.choice(len(POPULATION), size=cells, p=p)
    return scipy.sparse.vstack(
        [c[p][random.randint(0, c[p].shape[0] - 1), :] for p in pop]
    ).transpose()


def create_interaction(disease, c, variation):
    """Creates patient with interactions between cell types."""
    cells = 1000

    def r(low, high):
        return variation * random.uniform(low, high) + (1 - variation) * (high + low) / 2.

    p = np.zeros(len(POPULATION))
    p[0] = r(4, 69)  # bcell
    p[1] = r(0, 49)  # mono
    p[2] = r(0, 50)
    p[3] = r(22, 93) # cd4
    p[4] = r(0, 59)  # nk cells
    p[5] = r(0, 50)
    p[6] = r(0, 50)
    p[7] = r(6, 65)  # CD8 cytotoxic
    p[8] = r(0, 50)
    p[9] = r(0, 50)

    if disease:
        p[1] = random.uniform(49, 69)
        p[3] = random.uniform(93, 115)
    else:
        r = random.random()
        if r < 1. / 3:
            p[1] = random.uniform(49, 69)
        elif r < 2. / 3:
            p[3] = random.uniform(93, 115)

    p /= sum(p)

    pop = np.random.choice(len(POPULATION), size=cells, p=p)
    return scipy.sparse.vstack(
                [c[p][random.randint(0, c[p].shape[0] - 1), :] for p in pop]).transpose()


if __name__ == "__main__":
    main()

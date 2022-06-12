#!/usr/bin/env python3

"""Process lupus data into numpy."""

import math
import os
import pathlib

import numpy as np
import pandas
import scipy.sparse
import tqdm

import scanpy.api


if __name__ == "__main__":
    data = scanpy.api.read_h5ad(os.path.join("data", "SLEcrossX_nonorm.h5ad"))
    meta = pandas.read_csv(
        os.path.join("data", "CLUESImmVar_processed.V6_joined_pivot_cg_perc.txt"), sep=",")

    # Lupus vs healthy
    for state in set(data.obs["disease_cov"]):
        pathlib.Path(os.path.join("data", "lupus", state)).mkdir(parents=True, exist_ok=True)

    for ind in tqdm.tqdm(set(data.obs["ind_cov"])):
        state = list(set(data.obs[data.obs["ind_cov"] == ind]["disease_cov"]))
        assert(len(state) == 1)
        state = state[0]

        X = data.X[(data.obs["ind_cov"] == ind).values, :].transpose()

        scipy.sparse.save_npz(os.path.join("data", "lupus", state, "{}.npz".format(ind)), X)
        np.save(os.path.join("data", "lupus", state, "ct_{}.npy".format(ind)),
                data.obs[data.obs["ind_cov"] == ind]["ct_cov"].values)

    # Population (race)
    for state in set(data.obs["pop_cov"]):
        pathlib.Path(os.path.join("data", "lupus_pop", state)).mkdir(parents=True, exist_ok=True)

    for ind in tqdm.tqdm(set(data.obs["ind_cov"])):
        state = list(set(data.obs[data.obs["ind_cov"] == ind]["pop_cov"]))
        assert(len(state) == 1)
        state = state[0]

        X = data.X[(data.obs["ind_cov"] == ind).values, :].transpose()

        scipy.sparse.save_npz(os.path.join("data", "lupus_pop", state, "{}.npz".format(ind)), X)
        np.save(os.path.join("data", "lupus_pop", state, "ct_{}.npy".format(ind)),
                data.obs[data.obs["ind_cov"] == ind]["ct_cov"].values)


    # Monocyte
    pathlib.Path(os.path.join("data", "mono", "regression")).mkdir(parents=True, exist_ok=True)
    for ind in tqdm.tqdm(set(data.obs["ind_cov"])):
        X = data.X[(data.obs["ind_cov"] == ind).values, :].transpose()
        i = ind.split("_")[0]

        mono = meta[meta["ind_cov"] == i]["mono"].values.mean()
        if not math.isnan(mono):
            scipy.sparse.save_npz(
                os.path.join("data", "mono", "regression", "{}.npz".format(ind)), X)
            np.save(os.path.join("data", "mono", "regression", "ct_{}.npy".format(ind)),
                    data.obs[data.obs["ind_cov"] == ind]["ct_cov"].values)
            np.save(os.path.join("data", "mono", "regression", "value_{}.npy".format(ind)),
                    np.array(mono / 10))

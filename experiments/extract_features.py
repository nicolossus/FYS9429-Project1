#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import efel
import h5py
import numpy as np
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

infile = f"hh_sim_data_{rank}.h5"
outfile = f"hh_feature_data_{rank}.csv"

base_path = Path(__file__).resolve().parent
infile_path = base_path / "hh_data" / infile
outfile_path = base_path / "hh_data" / outfile

with open("features.json", "r") as f:
    features = json.load(f)["features"]

dfs = []

with h5py.File(infile_path, "r") as f:
    for grp_name in f.keys():
        gkbar = f[grp_name].attrs["gkbar"]
        gnabar = f[grp_name].attrs["gnabar"]
        glbar = f[grp_name].attrs["glbar"]
        el = f[grp_name].attrs["el"]
        t_sim = f[grp_name].attrs["t_sim"]
        dt = f[grp_name].attrs["dt"]
        stim_start = f[grp_name].attrs["stim_start"]
        stim_end = f[grp_name].attrs["stim_end"]

        v = f[grp_name]["v"][:]
        N = int(t_sim / dt) + 1
        t = np.linspace(0, t_sim, N)

        vtrace = {
            "T": t,
            "V": v,
            "stim_start": [stim_start],
            "stim_end": [stim_end],
        }

        # set the spike detection threshold in the eFEL, default -20.0
        # efel.api.set_threshold(-20.0)
        # set eFEL interpolation step
        efel.set_setting("interp_step", dt)

        sum_stats = efel.get_mean_feature_values([vtrace], features, raise_warnings=False)

        df_tmp = pd.DataFrame(sum_stats)
        df_tmp.insert(0, "gkbar", gkbar)
        df_tmp.insert(1, "gnabar", gnabar)
        df_tmp.replace({None: np.nan}, inplace=True)
        dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)
df.to_csv(outfile_path, index=False)

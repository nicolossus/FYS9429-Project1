#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run: mpirun -np 10 python run_hh_simulator.py
"""

from pathlib import Path

import numpy as np
import scipy.stats as stats
from mpi4py import MPI

from neurovae import hh_simulator

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


fname = f"hh_sim_data_{rank}.h5"
base_path = Path(__file__).resolve().parent
outfile = base_path / "hh_data" / fname

# priors
# gkbar: U(26, 46)
# gnabar: U(110, 130)
prior_min = np.array([26.0, 110.0])
prior_max = np.array([46.0, 130.0])
scale = prior_max - prior_min
prior = stats.uniform(loc=prior_min, scale=scale)

n_sims = 1000

for sim_i in range(n_sims):
    theta = prior.rvs()
    hh_simulator(
        gkbar=theta[0],
        gnabar=theta[1],
        glbar=0.3,
        el=-54.3,
        t_presim=200.0,
        t_sim=100.0,
        dt=0.01,
        stim_start=10.0,
        stim_end=60.0,
        stim_amp=15.0,
        outfile=outfile,
        sim_id=sim_i,
    )

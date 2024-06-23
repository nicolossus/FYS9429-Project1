#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from helper import fig_path

sns.set_theme(context="paper", style="darkgrid", rc={"axes.facecolor": "0.96"})
fontsize = "x-large"
params = {
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern"],
    "axes.labelsize": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "legend.handlelength": 2,
}
plt.rcParams.update(params)
plt.rc("text", usetex=True)


batches_len = 386
epochs = 100
T = batches_len * epochs
init_lr = 1e-3
alpha = 1e-2
p = 1
t = np.arange(0, T)

cosine_decay = 0.5 * (1 + np.cos(np.pi * t / T))
decayed = (1 - alpha) * cosine_decay**p + alpha
lr_schedule = init_lr * decayed

fig, ax = plt.subplots(tight_layout=True)
ax.plot(t, lr_schedule)
ax.set(xlabel="Iteration", ylabel="Learning rate")
plt.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

fig.savefig(fig_path("lr_schedule.pdf"), bbox_inches="tight")

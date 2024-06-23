#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from helper import fig_path

from neurovae import plot_digit, plot_digits


def plot_comparison(imgs, figsize=(5, 5), cmap="gray"):

    font = {
        "family": "serif",
        "color": "darkred",
        "weight": "normal",
        "size": 16,
    }

    dim = 4
    gridspec_kw = {"hspace": 0.5, "wspace": 0.05}
    fig, axes = plt.subplots(dim, dim, figsize=figsize, gridspec_kw=gridspec_kw)

    for n in range(dim**2):
        img = imgs[n]
        row_idx = n // dim
        col_idx = n % dim
        axi = axes[row_idx, col_idx]
        plot_digit(img, ax=axi, cmap=cmap)

    plt.figtext(0.5, 0.92, "Original images", ha="center", va="center", fontdict=font)
    plt.figtext(0.5, 0.5, "Reconstructed images", ha="center", va="center", fontdict=font)

    return fig, axes


def save_comparison(imgs, outfile, **kwargs):
    fig, _ = plot_comparison(imgs, **kwargs)
    fig.savefig(fig_path(outfile), bbox_inches="tight")


def save_samples(samples, outfile, **kwargs):
    fig, _ = plot_digits(samples, **kwargs)
    fig.savefig(fig_path(outfile), bbox_inches="tight")

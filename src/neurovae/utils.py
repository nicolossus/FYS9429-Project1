#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def plot_digit(img, label=None, ax=None, cmap="gray"):
    """Plot MNIST digit."""

    if ax is None:
        fig, ax = plt.subplots()

    if img.ndim == 1:
        img = img.reshape(28, 28)

    ax.imshow(img.squeeze(), cmap=cmap)
    ax.axis("off")
    if label is not None:
        ax.set_title(f"Label: {label}")

    return ax


def plot_digits(imgs, labels=None, dim=None, axes=None, figsize=(5, 5), cmap="gray"):
    """Plot MNIST digits on grid"""

    if dim is None:
        if axes is None:
            n_imgs = len(imgs)
            dim = np.sqrt(n_imgs)
            if not dim.is_integer():
                raise ValueError("If dim not specified `len(imgs)` must be a square number.")
            else:
                dim = int(dim)
        else:
            dim = len(axes)

    if axes is None:
        gridspec_kw = {"hspace": 0.05, "wspace": 0.05}
        if labels is not None:
            gridspec_kw["hspace"] = 0.25
        fig, axes = plt.subplots(dim, dim, figsize=figsize, gridspec_kw=gridspec_kw)

    for n in range(dim**2):
        img = imgs[n]
        row_idx = n // dim
        col_idx = n % dim
        axi = axes[row_idx, col_idx]
        if labels is not None:
            ax_label = labels[n]
        else:
            ax_label = None
        plot_digit(img, ax=axi, label=ax_label, cmap=cmap)

    return fig, axes

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from helper import fig_path

from neurovae import load_mnist, plot_digits

train_imgs, train_labels, _, _ = load_mnist(
    batch_size=None,
    drop_remainder=False,
    as_supervised=True,
    select_digits=list(range(4)),
    binarized=True,
    shuffle=False,
    shuffle_seed=None,
)

fig, _ = plot_digits(train_imgs[:16], train_labels[:16], cmap="gray")

fig.savefig(fig_path("bmnist.pdf"), bbox_inches="tight")

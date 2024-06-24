#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2024
#
# This file is part of neurovae
# SPDX-License-Identifier:    MIT

import importlib.metadata

from .conv_1d_vae import Conv1DVAE
from .conv_2d_vae import Conv2DVAE
from .datasets_loader import load_mnist
from .dnn_vae import DNNVAE
from .dnn_vae_asym import AsymDNNVAE
from .hh_simulator import hh_simulator
from .mlp_vae import MLPVAE
from .utils import plot_digit, plot_digits
from .vae_utils import (
    bce_loss,
    gaussian_kld,
    gaussian_sample,
    mse_loss,
    reparameterize,
    sse_loss,
)

__version__ = importlib.metadata.version(__package__)

__all__ = [
    "MLPVAE",
    "DNNVAE",
    "AsymDNNVAE",
    "Conv2DVAE",
    "Conv1DVAE",
    "load_mnist",
    "plot_digit",
    "plot_digits",
    "hh_simulator",
    "reparameterize",
    "gaussian_kld",
    "bce_loss",
    "mse_loss",
    "sse_loss",
    "gaussian_sample",
]

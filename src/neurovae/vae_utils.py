#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import optax
from jax import random


def reparameterize(rng, mean, logvar):
    return gaussian_sample(rng, mean, logvar)


def gaussian_sample(rng, mean, logvar):
    """Sample a diagonal Gaussian."""
    std = jnp.exp(0.5 * logvar)
    return mean + std * random.normal(rng, logvar.shape)


@jax.vmap
def gaussian_kld(mean, logvar):
    """KL divergence from a diagonal Gaussian to the standard Gaussian."""
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def bce_loss(recon_x, x):
    "Binary cross entropy with logits."
    return optax.sigmoid_binary_cross_entropy(recon_x, x).sum()


@jax.vmap
def sse_loss(recon_x, x):
    """Sum of squared errors"""
    return optax.squared_error(recon_x, x).sum()


@jax.vmap
def mse_loss(recon_x, x):
    """Mean squared error"""
    return optax.squared_error(recon_x, x).mean()

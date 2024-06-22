#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable

import flax.linen as nn

from .vae_utils import reparameterize


class MLPEncoder(nn.Module):
    """MLP VAE Encoder."""

    hidden_dim: int
    latent_dim: int
    activation_fn: Callable
    weight_init: Callable

    @nn.compact
    def __call__(self, x):
        x = self.activation_fn(nn.Dense(self.hidden_dim, kernel_init=self.weight_init())(x))
        mean = nn.Dense(self.latent_dim, name="mean")(x)
        logvar = nn.Dense(self.latent_dim, name="logvar")(x)
        return mean, logvar


class MLPDecoder(nn.Module):
    """MLP VAE Decoder."""

    hidden_dim: int
    output_dim: int
    activation_fn: Callable
    weight_init: Callable

    @nn.compact
    def __call__(self, z):
        z = self.activation_fn(nn.Dense(self.hidden_dim, kernel_init=self.weight_init())(z))
        recon_x = nn.Dense(self.output_dim)(z)
        return recon_x


class MLPVAE(nn.Module):
    """Full MLP VAE model."""

    hidden_dim: int = 500
    latent_dim: int = 20
    output_dim: int = 784
    activation_fn: Callable = nn.relu
    weight_init: Callable = nn.initializers.he_normal

    def setup(self):
        self.encoder = MLPEncoder(
            self.hidden_dim,
            self.latent_dim,
            self.activation_fn,
            self.weight_init,
        )
        self.decoder = MLPDecoder(
            self.hidden_dim,
            self.output_dim,
            self.activation_fn,
            self.weight_init,
        )

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z, assumption="bernoulli"):
        """Generate sample.

        The assumption of the data distribution determines the output activation
        function (in combination with the loss function):
            * Bernoulli assumption = logistic sigmoid + binary cross-entropy
            * Gaussian assumption = identity function + mean squared error
        """
        match assumption:
            case "bernoulli":
                return nn.sigmoid(self.decoder(z))
            case "gaussian":
                return self.decoder(z)
            case _:
                raise ValueError("Data distribution assumption not supported")

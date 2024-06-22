#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Sequence

import flax.linen as nn

from .vae_utils import reparameterize


class DNNEncoder(nn.Module):
    """DNN VAE Encoder."""

    hidden_dims: Sequence[int]
    latent_dim: int
    activation_fn: Callable
    weight_init: Callable

    @nn.compact
    def __call__(self, x):
        for hidden_dim in self.hidden_dims:
            x = self.activation_fn(nn.Dense(hidden_dim, kernel_init=self.weight_init())(x))
        mu = nn.Dense(self.latent_dim, name="mu")(x)
        logvar = nn.Dense(self.latent_dim, name="logvar")(x)
        return mu, logvar


class DNNDecoder(nn.Module):
    """DNN VAE Decoder."""

    hidden_dims: Sequence[int]
    output_dim: int
    activation_fn: Callable
    weight_init: Callable

    @nn.compact
    def __call__(self, z):

        for hidden_dim in reversed(self.hidden_dims):
            z = self.activation_fn(nn.Dense(hidden_dim, kernel_init=self.weight_init())(z))

        recon_x = nn.Dense(self.output_dim, name="recon_x")(z)
        return recon_x


class DNNVAE(nn.Module):
    """Full DNN VAE model."""

    hidden_dims: Sequence = (500, 150)
    latent_dim: int = 20
    output_dim: int = 784
    activation_fn: Callable = nn.relu
    weight_init: Callable = nn.initializers.he_normal

    def setup(self):
        self.encoder = DNNEncoder(
            self.hidden_dims,
            self.latent_dim,
            self.activation_fn,
            self.weight_init,
        )
        self.decoder = DNNDecoder(
            self.hidden_dims,
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

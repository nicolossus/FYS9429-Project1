#!/usr/bin/env python
# -*- coding: utf-8 -*-

import flax.linen as nn

from .vae_utils import reparameterize


class Conv1DEncoder(nn.Module):
    """1D convolutional VAE encoder."""

    latent_dim: int

    @nn.compact
    def __call__(self, x):

        x = nn.Conv(
            features=32,
            kernel_size=(3,),
            strides=1,
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=64,
            kernel_size=(3,),
            strides=1,
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # flatten image grid to single feature vector

        # x = nn.Dense(features=256, kernel_init=nn.initializers.he_normal())(x)
        # x = nn.relu(x)

        mu = nn.Dense(features=self.latent_dim)(x)
        logvar = nn.Dense(features=self.latent_dim)(x)
        return mu, logvar


class Conv1DDecoder(nn.Module):
    """1D convolutional VAE decoder."""

    output_dim: int

    @nn.compact
    def __call__(self, z):

        z = nn.Dense(features=self.output_dim * 64, kernel_init=nn.initializers.he_normal())(z)
        z = nn.relu(z)
        z = z.reshape(z.shape[0], self.output_dim, 64)

        # 7x7x64 => 14x14x64
        z = nn.ConvTranspose(
            features=64,
            kernel_size=(3,),
            strides=1,
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
        )(z)
        z = nn.relu(z)

        z = nn.ConvTranspose(
            features=32,
            kernel_size=(3,),
            strides=1,
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
        )(z)
        z = nn.relu(z)

        # no activation
        recon_x = nn.ConvTranspose(
            features=1,
            kernel_size=(3,),
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
        )(z)

        recon_x = recon_x.reshape((recon_x.shape[0], -1))  # flatten

        return recon_x


class Conv1DVAE(nn.Module):
    """Full 2D convolutional VAE model."""

    latent_dim: int = 20
    output_dim: int = 10000

    def setup(self):
        self.encoder = Conv1DEncoder(self.latent_dim)
        self.decoder = Conv1DDecoder(self.output_dim)

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

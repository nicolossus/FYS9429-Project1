#!/usr/bin/env python
# -*- coding: utf-8 -*-

import flax.linen as nn

from .vae_utils import reparameterize


class Conv2DEncoder(nn.Module):
    """2D convolutional VAE encoder."""

    latent_dim: int

    @nn.compact
    def __call__(self, x):

        # 28x28x1 => 14x14x32
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=2,
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = nn.relu(x)

        # 14x14x32 => 7x7x64
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=2,
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # flatten image grid to single feature vector
        x = nn.Dense(features=256, kernel_init=nn.initializers.he_normal())(x)
        x = nn.relu(x)

        mu = nn.Dense(features=self.latent_dim)(x)
        logvar = nn.Dense(features=self.latent_dim)(x)
        return mu, logvar


class Conv2DDecoder(nn.Module):
    """2D convolutional VAE decoder."""

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(features=7 * 7 * 64, kernel_init=nn.initializers.he_normal())(z)
        z = nn.relu(z)
        z = z.reshape(z.shape[0], 7, 7, 64)

        # 7x7x64 => 14x14x64
        z = nn.ConvTranspose(
            features=64,
            kernel_size=(3, 3),
            strides=2,
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
        )(z)
        z = nn.relu(z)

        # 14x14x64 => 28x28x32
        z = nn.ConvTranspose(
            features=32,
            kernel_size=(3, 3),
            strides=2,
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
        )(z)
        z = nn.relu(z)

        # 28x28x32 => 28x28x1
        # no activation
        recon_x = nn.ConvTranspose(
            features=1,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
        )(z)

        return recon_x


class Conv2DVAE(nn.Module):
    """Full 2D convolutional VAE model."""

    latent_dim: int = 20

    def setup(self):
        self.encoder = Conv2DEncoder(self.latent_dim)
        self.decoder = Conv2DDecoder()

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

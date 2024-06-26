import flax.linen as nn
import jax
import jax.numpy as jnp


class Conv1DEncoder(nn.Module):
    """1D convolutional VAE encoder."""

    latent_dim: int

    @nn.compact
    def __call__(self, x):

        x = nn.Conv(
            features=32,
            kernel_size=(3,),
            strides=2,
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=64,
            kernel_size=(3,),
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


tabulate_fn = nn.tabulate(Conv1DEncoder(20), jax.random.key(0), compute_flops=True, compute_vjp_flops=True)
x = jnp.ones((64, 10000))
print(tabulate_fn(x))

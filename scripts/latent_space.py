"""
file: customs/latent_space.py.

This module provides a function to generate latent space points for a VAE-GAN model.
"""

import tensorflow as tf
from tensorflow import Tensor


@tf.function
def generate_latent_points(
    latent_dimensions: Tensor,
    n: int,
    std: Tensor = 1,
    turn_on_reproducibility: bool = False,
    number_generator: tf.random.Generator = None,
) -> Tensor:
    """
    Generate latent space points to be used as input for the generator.
    Drawn from a Gaussian distribution with mean 0 and standard deviation 1.

    Args:
        latent_dimensions (int): Dimension of the latent space.
        n (int): Number of samples to generate.
        number_generator (tf.random.Generator): Random generator for reproducibility.
        turn_on_reproducibility (bool): Whether to turn on reproducibility.

    Returns:
        Tensor: Tensor of shape (n, 1, latent_dimensions) containing the latent space points.
    """
    if turn_on_reproducibility:
        return number_generator.normal(shape=(n, latent_dimensions), stddev=std)
    else:
        return tf.random.normal(shape=(n, latent_dimensions), stddev=std)

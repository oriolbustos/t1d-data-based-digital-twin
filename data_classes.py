from dataclasses import dataclass

import numpy as np
import tensorflow as tf

@dataclass
class TrainingData:
    """
    Data class containing the simulation dynamical inputs, results and some minor settings.
    """

    scBG: np.ndarray  # Real scaled blood glucose profile (single vector)

    scPI: np.ndarray = None  # Scaled Plasma Insulin   AND PACKED
    scRA: np.ndarray = None  # Scaled Rate of Appearance AND PACKED

    # Dynamical input to the model
    X_real_BG: tf.Tensor = None  # Batch of real blood glucose samples
    X_fake_BG: tf.Tensor = None  # Batch of generated blood glucose samples

    Y_real_label: tf.Tensor = None  # Label for real samples
    Y_fake_label: tf.Tensor = None  # Label for generated samples

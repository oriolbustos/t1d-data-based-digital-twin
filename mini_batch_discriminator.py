"""
file: customs/mini_batch_discriminator.py.

This file contains the implementation of a mini-batch discrimination layer
to enhance the diversity of generated samples in Generative Adversarial Networks (GANs).
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


# Mini-batch discrimination layer
class MinibatchDiscrimination(Layer):
    """
    Mini-batch discrimination layer to encourage diversity in GANs by comparing features across samples in a batch.
    """

    def __init__(self, num_kernels, dim_per_kernel):
        """
        Initialize the MinibatchDiscrimination layer.

        Args:
            num_kernels (int): Number of kernels to use for discrimination.
            dim_per_kernel (int): Dimension per kernel.
        """
        super(MinibatchDiscrimination, self).__init__()  # noqa: UP008
        self.num_kernels = num_kernels
        self.dim_per_kernel = dim_per_kernel

    def build(self, input_shape):
        """
        Create the trainable weight matrix for the MinibatchDiscrimination layer.

        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        self.T = self.add_weight(
            name='T',
            shape=(input_shape[-1], self.num_kernels * self.dim_per_kernel),
            initializer='uniform',
            trainable=True,
        )

    def call(self, x):
        """
        Apply the mini-batch discrimination operation to the input tensor.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Concatenated tensor with mini-batch features.
        """
        m = tf.matmul(x, self.T)
        m = tf.reshape(m, [-1, self.num_kernels, self.dim_per_kernel])
        diffs = tf.expand_dims(m, 3) - tf.expand_dims(tf.transpose(m, perm=[1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), axis=2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), axis=2)
        return tf.concat([x, minibatch_features], axis=1)

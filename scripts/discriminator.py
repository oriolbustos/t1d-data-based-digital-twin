"""
file: vae/discriminator.py.

This file contains the Discriminator class, which is a subclass of the Keras Model class.
It is used to build the discriminator model of the GAN model.
"""


import tensorflow as tf
from clip_constraint import ClipConstraint
from loss_functions import wasserstein_loss
from mini_batch_discriminator import MinibatchDiscrimination
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import LSTM, BatchNormalization, Concatenate, Dense, Flatten, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from types import SimpleNamespace


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()  # noqa: UP008
        self.settings = SimpleNamespace()
        self.settings.weight_clip = 1
        self.settings.in_data_shape = (1, 1)
        self.settings.out_data_shape = (1, 18)
        self.settings.gan_inputs = ['BG', 'PI', 'RA']
        self.settings.learning_rate_discriminator = 0.0001


        self._model = self.build_discriminator()
        self.compile_model()  # discriminator is compiled here, generator only in the composite GAN model

    def build_discriminator(self):
        """
        Build the discriminator model architecture.

        Returns:
            Model: The compiled Keras Model for the discriminator.
        """
        init = RandomNormal(stddev=0.02)
        const = ClipConstraint(self.settings.weight_clip)

        input_bg = Input(shape=self.settings.out_data_shape, name='in_BG_disc')

        inputs = {}
        processed_inputs = []
        for input_name in self.settings.gan_inputs:
            if input_name != 'BG':
                inputs[input_name] = Input(shape=self.settings.in_data_shape, dtype=tf.float32, name=f'input_{input_name}_disc')
                processed_inputs.append(inputs[input_name])

        merged = Concatenate()(processed_inputs + [input_bg])  # noqa: RUF005
        d = LSTM(128, kernel_initializer=init, kernel_constraint=const)(merged)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Flatten()(d)
        d = MinibatchDiscrimination(num_kernels=100, dim_per_kernel=5)(d)
        output = Dense(1)(d)

        d_model = Model([processed_inputs, input_bg], outputs=output)
        return d_model

    def compile_model(self) -> None:
        """
        Compile the discriminator model with optimizer, loss, and metrics.
        """
        self.compile(
            optimizer=RMSprop(learning_rate=self.settings.learning_rate_discriminator),
            loss=wasserstein_loss,
            metrics=['accuracy'],
        )

    @tf.function
    def call(self, inputs, training=None, mask=None):
        """
        Discriminator call method.
        Expects the inputs to be a list of n elements. One for each input layer.
        """
        return self._model(inputs)

    def get_functional_model(self) -> Model:
        """Return the functional Keras model."""
        return self._model

    @property
    def input(self):
        """Return the model's input."""
        return self._model.input

    @property
    def output(self):
        """Return the model's output."""
        return self._model.output

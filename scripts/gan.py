"""
file: vae/gan.py.

This file contains the GANModel class, which is responsible for training a GAN model
using a pre-trained VAE generator and a discriminator. The class handles the loading of the generator model,
the definition of the GAN model, and the training process, including the discriminator and generator loops.
"""

from pathlib import Path

import tensorflow as tf
from data_classes import TrainingData
from latent_space import generate_latent_points
from loss_functions import wasserstein_loss
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from discriminator import Discriminator
from types import SimpleNamespace

class GANModel:

    def __init__(self, settings: SimpleNamespace, decoder_model: Model):

        self.settings = settings
        self.g_model = decoder_model
        self.d_model = Discriminator(settings)
        self.gan_model = self.define_gan()

        self.number_generator = tf.random.Generator.from_seed(1)

    def load_generator_model(self, filename) -> Model:
        """
        Load the pre-trained VAE generator model from the specified file.
        Args:
            filename (Path): Path to the pre-trained VAE generator model file.
        Returns:
            Model: The loaded VAE generator model.
        """
        return load_model(filename, compile=False)

    def define_gan(self):
        """
        Defines and compiles the GAN model by combining the generator and discriminator models.

        Returns:
            Model: The compiled GAN model.
        """
        for layer in self.d_model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

        generator_input = self.get_generator_input()

        generator_output = self.g_model(generator_input)

        condition_inputs = generator_input[1:]
        latent_input = generator_input[0]
        discriminator_input = condition_inputs + [generator_output]  # noqa: RUF005
        discriminator_output = self.d_model(discriminator_input)

        gan_input = [condition_inputs, latent_input]
        gan_model = Model(inputs=gan_input, outputs=[discriminator_output, generator_output])
        opt = Adam(learning_rate=self.settings.learning_rate_generator)
        gan_model.compile(loss=[wasserstein_loss, 'mse'], loss_weights=self.settings.ratio_generator_losses, optimizer=opt)
        return gan_model

    def train(self, patient_data: TrainingData):
        self.settings.total_steps_gan = (len(patient_data.scPI) // self.settings.batch_size_gan) * self.settings.n_epochs_gan
        history = {"step": [], "d_loss_real": [], "d_loss_fake": [], "g_loss": [], "g_adversarial": [], "g_l2": []}

        for i in tqdm(range(self.settings.total_steps_gan), total=self.settings.total_steps_gan, desc="GAN Training", unit="step"):

            for _ in range(self.settings.n_critic):
                d_real, d_fake = self.discriminator_loop(patient_data)
            g_total, g_adv, g_l2 = self.generator_loop(patient_data)

            history["step"].append(i + 1)
            history["d_loss_real"].append(float(d_real))
            history["d_loss_fake"].append(float(d_fake))
            history["g_loss"].append(float(g_total))
            history["g_adversarial"].append(float(g_adv))
            history["g_l2"].append(float(g_l2))

        self.settings.total_steps_gan = i
        return history

    def discriminator_loop(self, patient_data: TrainingData):
        x_real_bg, condition_inputs, y_real_label = self.generate_real_samples(patient_data, self.settings.batch_size_gan)
        x_fake_bg, y_fake_label = self.generate_fake_samples(condition_inputs)
        d_loss_real, _ = self.d_model.train_on_batch([condition_inputs, x_real_bg], y_real_label)
        d_loss_fake, _ = self.d_model.train_on_batch([condition_inputs, x_fake_bg], y_fake_label)
        return d_loss_real, d_loss_fake

    def generator_loop(self, patient_data: TrainingData):
        latent_input = generate_latent_points(self.settings.latent_dimensions, self.settings.batch_size_gan)
        x_real_bg, condition_inputs, y_real_label = self.generate_real_samples(patient_data, self.settings.batch_size_gan)
        gan_inputs = list(condition_inputs) + [latent_input]
        gan_labels = [y_real_label, x_real_bg]
        g_loss, adversarial_loss, l2_loss = self.gan_model.train_on_batch(gan_inputs, gan_labels)
        return g_loss, adversarial_loss, l2_loss


    def generate_real_samples(self, dataset, batch_size):
        """
        Generates real samples from the dataset for training the discriminator.

        Args:
            dataset: The dataset containing real samples.
            batch_size: The number of samples to generate.

        Returns:
            Tuple containing real BG input, conditional inputs, and real labels.
        """
        total_samples = tf.shape(dataset.scBG)[0]
        ix = self.number_generator.uniform((batch_size,), minval=0, maxval=total_samples, dtype=tf.int32)
        # ix = tf.random.uniform((batch_size,), minval=0, maxval=total_samples, dtype=tf.int32)

        conditional_inputs = []
        bg_real_input = tf.gather(dataset.scBG, ix)[:, :, : self.settings.glucose_dim]
        for input_name in self.settings.gan_inputs:
            if input_name != 'BG':
                conditional_inputs.append(tf.gather(getattr(dataset, f'sc{input_name}'), ix))
                y_real_label = tf.fill((batch_size, 1, 1), self.settings.y_real_label)

        return bg_real_input, conditional_inputs, y_real_label

    def generate_fake_samples(self, condition_inputs):
        """
        Generates fake samples using the generator model and provided condition inputs.

        Args:
            condition_inputs: List of conditional inputs for the generator.

        Returns:
            Tuple containing generated fake BG input and fake labels.
        """
        latent_input = generate_latent_points(self.settings.latent_dimensions, self.settings.batch_size_gan)

        generator_inputs = [latent_input] + condition_inputs  # noqa: RUF005
        g_output = self.g_model(generator_inputs, training=False)

        x_fake_bg = g_output
        y_fake_label = tf.fill((self.settings.batch_size_gan, 1, 1), self.settings.y_fake_label)

        return x_fake_bg, y_fake_label

    def get_generator_input(self):
        """
        Get the input layers of the generator model.
        This method creates new Input layers based on the input shape of the generator model.
        """

        def flatten(inputs):
            """Helper function to flatten the input list."""
            flat = []
            for item in inputs:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            return flat

        # Ensure g_model.input is a flat list of tensors
        g_model_inputs = self.g_model.input if isinstance(self.g_model.input, list) else [self.g_model.input]
        g_model_inputs = flatten(g_model_inputs)

        # Create new Input layers based on the flattened inputs
        generator_input = [Input(shape=input_tensor.shape[1:], name=input_tensor.name.split(':')[0]) for input_tensor in g_model_inputs]

        if not generator_input[0].name.startswith('in_lat'):
            raise ValueError("Last input is not 'in_lat_gen'")

        return generator_input

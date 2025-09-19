"""
file: vae/simulate_vae_gan.py.

This module contains the SimulatorVAEGAN class, which is responsible for simulating blood glucose data
using a Variational Autoencoder (VAE) or Generative Adversarial Network (GAN) model.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from utils import unscale_data
from tensorflow import Tensor
from utils import apply_transformation_params
from types import SimpleNamespace
from tqdm import tqdm

class SimulatorVAEGAN:
    def __init__(self):
        self.sim_data = None

        self.settings = SimpleNamespace()
        self.settings.gan_inputs = ['BG', 'PI', 'RA']
        self.settings.glucose_dim = int(90/5) # prediction horizon 90 min, data every 5 min

    def aggregate_arrays(self, shifted_arrays: np.ndarray, time_steps: int, padding_steps: int = 0) -> np.ndarray:
        params = {
            'padding_steps': padding_steps,
            'time_steps': time_steps,
        }

        # compute
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            output_profile = np.nanmean(shifted_arrays, axis=1)[0, params['padding_steps'] : params['time_steps'] + params['padding_steps']]
            std_gen_bg_scaled = np.nanstd(shifted_arrays, axis=1)  # get the std of the arrays

        # check output_profile has shape (time_steps,)
        if output_profile.shape[0] != time_steps:
            raise ValueError(f'Output profile shape mismatch: expected {time_steps}, got {output_profile.shape[0]}')

        std_output_profile = std_gen_bg_scaled[
            0, padding_steps : time_steps + padding_steps
        ]  # take until time_steps, as further values are not filled

        return output_profile, std_output_profile

    def prepare_curve_inputs(self, t_step: Tensor, sim_data, latent_inputs: np.ndarray) -> list:
        """
        Prepare the inputs for the GAN model at a specific time step.
        Args:
            t_step (Tensor): Current time step in the simulation.
            sim_data (SimulationData): Simulation data containing real and scaled blood glucose values.
            latent_inputs (np.ndarray): Latent inputs generated
                using a Wiener process.
        Returns:
            list: List of inputs for the GAN model, including latent inputs and other conditions.
        """
        inputs = []
        inputs.append(tf.convert_to_tensor([latent_inputs[t_step]], dtype=tf.float32))

        for input_name in self.settings.gan_inputs:
            if input_name != 'BG':
                input_data = tf.gather(getattr(sim_data, f'{input_name}_scaled'), t_step)
                inputs.append(input_data)
        return inputs

    def simulate(self, sim_data, generator, simulation_length):
        """
        Generate blood glucose data for a specific patient using their GAN model and calculate metrics.

        Args:
            patient_id (str): ID of the patient.
            gan_model_path (str): Path to the GAN model for the patient.
        """
        # Preprocess the patient's data
        self.generator = generator

        total_time_steps = tf.shape(sim_data.real_bg_unscaled)[0]
        simulation_steps = min(int(288 * simulation_length), int(total_time_steps))

        latent_dim = 18

        initial_latent = np.random.normal(loc=0, scale=1.0, size=(latent_dim,))
        latent_inputs = self.generate_latent_wiener_process(initial_latent, simulation_steps, latent_dim)

        for i in tqdm(range(simulation_steps-1)):
            inputs = self.prepare_curve_inputs(i, sim_data, latent_inputs)
            # latent_input, *condition_inputs = inputs[0], inputs[1:]

            gen_bg = self.generator(inputs, training=False)
            sim_data.gen_bg_scaled_arrays[0, i, i : i + self.settings.glucose_dim] = gen_bg

        shifted_arrays = np.where(
            sim_data.gen_bg_scaled_arrays == 0, np.nan, sim_data.gen_bg_scaled_arrays
        )  # replace 0 with nan so that they dont affect the next step

        gen_bg_profile_scaled, std_output_profile = self.aggregate_arrays(
            shifted_arrays, simulation_steps, 0
        )
        sim_data.gen_bg_unscaled = unscale_data(
            gen_bg_profile_scaled,
            'misc/',
        )
        sim_data.std_gen_bg_unscaled = unscale_data(
            std_output_profile,
            'misc/',
        )
        sim_data.gen_bg_scaled = gen_bg_profile_scaled

        return sim_data

    @staticmethod
    def generate_latent_wiener_process(initial_latent, n_steps, latent_dim, theta=0.1, mean=0.0, drift=0.0, diffusion=1.0):
        """
        Generate latent variables using a Wiener process.

        Args:
            initial_latent (array): Initial latent vector.
            n_steps (int): Number of time steps to generate.
            latent_dim (int): Dimension of the latent space.

        Returns:
            np.array: Latent variables over time.
        """
        latent_inputs = np.zeros((n_steps, latent_dim))
        latent_inputs[0] = initial_latent

        for t in range(1, n_steps):
            dw = np.random.normal(loc=drift, scale=np.sqrt(diffusion), size=(latent_dim,))
            latent_inputs[t] = latent_inputs[t - 1] + theta * (mean - latent_inputs[t - 1]) + dw * 1

        return latent_inputs

    def save_simulation_results(
        self,
        gen_bg_unscaled: np.ndarray,
        last_gen_bg_unscaled: np.ndarray,
        gen_bg_scaled: np.ndarray,
    ) -> None:
        """
        Save simulation results to a CSV file, for each simulated patient.
        The file will contain the generated blood glucose values only. (*provisional I may change this).

        Args:
            simulated_patient (int): Index of the patient being simulated.
            results_dir (str): Directory to save the results.
            gen_bg_unscaled (np.ndarray): Array of unscaled blood glucose values.
        """
        # Make sure they are of equal length
        length_to_use = min(len(gen_bg_unscaled), len(self.sim_data.real_bg_unscaled))

        if last_gen_bg_unscaled is None or len(last_gen_bg_unscaled) != len(gen_bg_unscaled):
            last_gen_bg_unscaled = np.zeros_like(gen_bg_unscaled)

        gen_bg_unscaled_transformed = apply_transformation_params(
            gen_bg_unscaled,
            self.settings.validation.scale_factor_simulated_data,
            self.settings.validation.offset_simulated_data,
        )

        results_df = pd.DataFrame(
            {
                'BG_gen': gen_bg_unscaled[:length_to_use],
                'BG_real': self.sim_data.real_bg_unscaled[:length_to_use],
                'BG_last_gen': last_gen_bg_unscaled[:length_to_use],
                'BG_gen_transformed': gen_bg_unscaled_transformed[:length_to_use],
                'BG_gen_scaled': gen_bg_scaled[:length_to_use],
                'BG_real_scaled': self.sim_data.real_bg_scaled[:length_to_use],
                'PI': np.squeeze(self.sim_data.PI_unscaled[:length_to_use]) if 'PI' in self.settings.gan_inputs else None,
                'RA': np.squeeze(self.sim_data.RA_unscaled[:length_to_use]) if 'RA' in self.settings.gan_inputs else None,
                'PA': np.squeeze(self.sim_data.PA_unscaled[:length_to_use]) if 'PA' in self.settings.gan_inputs else None,
            }
        )
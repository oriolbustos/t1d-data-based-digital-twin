import numpy as np
import os
from pathlib import Path
import joblib

def apply_transformation_params(generated_bg: np.ndarray, scale_factor_simulated_data: float, offset_simulated_data: float) -> np.ndarray:
    """Apply scaling to BG profiles if required."""
    generated_bg_scaled = (
        np.mean(generated_bg) + offset_simulated_data + scale_factor_simulated_data * (generated_bg - np.mean(generated_bg))
    )
    generated_bg_scaled = np.where(generated_bg_scaled < 5, 5, generated_bg_scaled)
    return generated_bg_scaled

def reverse_manual_minmax_scaling(scaled_data: np.array, min_val: float, max_val: float, scaler_range: tuple) -> np.array:
    """
    Reverse manual min-max scaling on the data.

    Args:
        scaled_data (np.array): Scaled data to reverse.
        min_val (float): Minimum value, saved when doing the original scaling.
        max_val (float): Maximum value, saved when doing the original scaling.
    Returns:
        np.array: Reversed scaled data.
    """
    lower_bound = scaler_range[0]
    upper_bound = scaler_range[1]

    unscaled_data = min_val + (scaled_data - lower_bound) * (max_val - min_val) / (upper_bound - lower_bound)
    return unscaled_data

def unscale_data(
    gen_bg_scaled: np.ndarray,
    data_path: Path,
    scaler_range: tuple=(0, 1),
) -> np.ndarray:
    # Load the saved scalers
    scalers_file: str = os.path.join(data_path, 'bg_scalers.joblib')
    scalers_df = joblib.load(scalers_file)

    original_min = scalers_df.iloc[0]['BG_min']
    original_max = scalers_df.iloc[0]['BG_max']
    unscaled_bg = reverse_manual_minmax_scaling(gen_bg_scaled, original_min, original_max, scaler_range)

    return np.round(unscaled_bg, 3)


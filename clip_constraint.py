"""
file: utils/clip_constraint.py.

This file contains the ClipConstraint class, a subclass of the Keras Constraint class.
"""

from keras import backend
from keras.constraints import Constraint


class ClipConstraint(Constraint):
    """
    ClipConstraint class, a subclass of the Keras Constraint class.
    Used to clip the weights of the discriminator model, so that they are within a certain range,
    and avoid big gradients that could lead to the vanishing gradient problem.
    """

    def __init__(self, clip_value):
        """Initialize the ClipConstraint with a clip value.
        Args:
            clip_value (float): The value to clip the weights to.
        """
        self.clip_value = clip_value

    def __call__(self, weights):
        """Call method to apply the constraint to the weights.
        Args:
            weights (tensor): The weights to be clipped.
        Returns:
            tensor: The clipped weights.
        """
        return backend.clip(weights, -self.clip_value, self.clip_value)

    def get_config(self) -> dict:
        """Get the configuration of the constraint.
        Returns:
            dict: A dictionary containing the configuration of the constraint.
        """
        return {'clip_value': self.clip_value}

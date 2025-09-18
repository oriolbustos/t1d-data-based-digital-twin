import tensorflow as tf


def wasserstein_loss(y_true, y_pred):
    """
    Wasserstein loss function.
    This function is used to compute the Wasserstein distance between the true and predicted values.

    Args:
        y_true (tf.Tensor): The true labels. These are -1.0 for real samples and 1.0 for generated samples.
                                        (As generated in generate_real_samples and generate_fake_samples.)
        y_pred (tf.Tensor): The predicted values from the discriminator.

    Returns:
        tf.Tensor: The computed Wasserstein loss, which is the mean of the
                   element-wise product of y_true and y_pred.

    Note:
        The loss is calculated as E[D(real)] - E[D(fake)], which simplifies to the mean of
        y_true * y_pred when y_true is properly set.
    """
    return -tf.reduce_mean(y_true * y_pred)
import tensorflow as tf
from tensorflow.keras import layers, Model


# TensorFlow 2 equivalent
def random_normal_variable(shape, mean, scale, name=None):
    """Creates a random normal variable.

  Args:
    shape: A tuple of integers, the shape of the tensor to be created.
    mean: A float, the mean value of the normal distribution.
    scale: A float, the standard deviation of the normal distribution.
    dtype: The data type of the tensor.
    name: Optional name for the operation.
    seed: Optional random seed.

  Returns:
    A tensor with the specified shape, filled with random values from a normal
    distribution.
  """

    # Create a random tensor with the specified shape and distribution
    initial_value = tf.random.normal(shape, mean=mean, stddev=scale)

    # Create a variable from the random tensor
    var = initial_value

    return var


# TensorFlow 2 equivalent
def in_train_phase(x, alt, learning_phase=True):
    """Returns either `x` or `alt` depending on whether we are in training phase.

    Args:
    x: The value to be returned in train phase.
    alt: The value to be returned in test phase.

    Returns:
    Either `x` or `alt` depending on the current phase.
    """

    if learning_phase == 1:
        return x
    else:
        return alt


def variational_layers(z_mean, enc, kl_loss_var, params, is_training=True):
    def sampling(args):
        z_mean, z_log_var = args

        epsilon = random_normal_variable(shape=(params['batch_size'], params['hidden_dim']),
                                                          mean=0., scale=1.)
        # insert kl loss here

        z_rand = z_mean + tf.exp(z_log_var / 2) * kl_loss_var * epsilon
        return in_train_phase(z_rand, z_mean,learning_phase=is_training)

    # variational encoding
    z_log_var_layer = layers.Dense(params['hidden_dim'], name='z_log_var_sample')
    z_log_var = z_log_var_layer(enc)

    z_mean_log_var_output = layers.Concatenate(
        name='z_mean_log_var')([z_mean, z_log_var])

    z_samp = layers.Lambda(sampling, z_mean.shape)([z_mean, z_log_var])

    if params['batchnorm_vae']:
        z_samp = layers.BatchNormalization(axis=-1)(z_samp)

    return z_samp, z_mean_log_var_output

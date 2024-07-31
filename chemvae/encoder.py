import tensorflow as tf
from tensorflow.keras import layers, Model


def encoder_model(params):
    x_in = layers.Input(shape=(params['MAX_LEN'], params['NCHARS']),
                        name='input_molecule_smi')

    x = layers.Conv1D(filters=int(params['conv_dim_depth'] * params['conv_d_growth_factor']),
                      kernel_size=int(params['conv_dim_width'] * params['conv_w_growth_factor']),
                      activation='tanh',
                      name="encoder_conv0")(x_in)

    if params['batchnorm_conv']:
        x = layers.BatchNormalization(axis=-1,
                                      name="encoder_norm0")(x)

    for j in range(1, params['conv_depth'] - 1):
        x = layers.Conv1D(filters=int(params['conv_dim_depth'] * params['conv_d_growth_factor'] ** (j)),
                          kernel_size=int(params['conv_dim_width'] * params['conv_w_growth_factor'] ** (j)),
                          activation='tanh',
                          name="encoder_conv{}".format(j))(x)

    if params['batchnorm_conv']:
        x = layers.BatchNormalization(axis=-1,
                                      name="encoder_norm{}".format(j))(x)

    x = layers.Flatten()(x)

    if params['middle_layer'] > 0:
        middle = layers.Dense(
            units=int(params['hidden_dim'] * params['hg_growth_factor'] ** (params['middle_layer'] - 1)),
            activation=params['activation'], name='encoder_dense0')(x)
        if params['dropout_rate_mid'] > 0:
            middle = layers.Dropout(rate=params['dropout_rate_mid'])(middle)
        if params['batchnorm_mid']:
            middle = layers.BatchNormalization(axis=-1, name='encoder_dense0_norm')(middle)

        for i in range(2, params['middle_layer'] + 1):
            middle = layers.Dense(
                units=int(params['hidden_dim'] * params['hg_growth_factor'] ** (params['middle_layer'] - i)),
                activation=params['activation'],
                name='encoder_dense{}'.format(i))(middle)
            if params['dropout_rate_mid'] > 0:
                middle = layers.Dropout(rate=params['dropout_rate_mid'])(middle)
            if params['batchnorm_mid']:
                middle = layers.BatchNormalization(axis=-1,
                                                   name='encoder_dense{}_norm'.format(i))(middle)
    else:
        middle = x

    z_mean = layers.Dense(params['hidden_dim'], name='z_mean_sample')(middle)

    encoder = Model(x_in, [z_mean, middle], name="encoder")

    return encoder

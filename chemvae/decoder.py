import tensorflow as tf
from tensorflow.keras import layers, Model


def decoder_model(params):
    z_in = layers.Input(shape=(params['hidden_dim'],), name='decoder_input')

    true_seq_in = layers.Input(shape=(params['MAX_LEN'], params['NCHARS']),
                               name='decoder_true_seq_input')

    z = layers.Dense(units=int(params['hidden_dim']),
                     activation=params['activation'],
                     name="decoder_dense0")(z_in)

    if params['dropout_rate_mid'] > 0:
        z = layers.Dropout(rate=params['dropout_rate_mid'])(z)

    if params['batchnorm_mid']:
        z = layers.BatchNormalization(axis=-1,
                                      name="decoder_dense0_norm")(z)

    for i in range(1, params['middle_layer']):
        z = layers.Dense(units=int(params['hidden_dim'] * params['hg_growth_factor'] ** (i)),
                         activation=params['activation'],
                         name="decoder_dense{}".format(i))(z)
        if params['dropout_rate_mid'] > 0:
            z = layers.Dropout(rate=params['dropout_rate_mid'])(z)

        if params['batchnorm_mid']:
            z = layers.BatchNormalization(axis=-1,
                                          name="decoder_dense{}_norm".format(i))(z)

    z_reps = layers.RepeatVector(n=params['MAX_LEN'])(z)

    if params['gru_depth'] > 1:
        x_dec = layers.GRU(units=params['recurrent_dim'],
                           return_sequences=True,
                           activation='tanh',
                           name='decoder_gru0')(z_reps)

        for k in range(params['gru_depth'] - 2):
            x_dec = layers.GRU(units=params['recurrent_dim'],
                               return_sequences=True,
                               activation='tanh',
                               name="decoder_gru{}".format(k + 1))(x_dec)

        if params['do_tgru']:
            x_out = TerminalGRU(params['NCHARS'],
                                rnd_seed=params['RAND_SEED'],
                                recurrent_dropout=params['tgru_dropout'],
                                return_sequences=True,
                                activation='softmax',
                                temperature=0.01,
                                name='decoder_tgru',
                                implementation=params['terminal_GRU_implementation'])([x_dec, true_seq_in])
        else:
            x_out = layers.GRU(units=params['NCHARS'],
                               return_sequences=True,
                               activation='softmax',
                               name='decoder_gru_final')(x_dec)
    else:
        if params['do_tgru']:
            x_out = TerminalGRU(params['NCHARS'],
                                rnd_seed=params['RAND_SEED'],
                                recurrent_dropout=params['tgru_dropout'],
                                return_sequences=True,
                                activation='softmax',
                                temperature=0.01,
                                name='decoder_tgru',
                                implementation=params['terminal_GRU_implementation'])([z_reps, true_seq_in])
        else:
            x_out = layers.GRU(units=params['NCHARS'],
                               return_sequences=True,
                               activation='softmax',
                               name='decoder_gru_final')(z_reps)

    if params['do_tgru']:
        decoder = Model([z_in, true_seq_in], x_out, name="decoder")
    else:
        decoder = Model(z_in, x_out, name="decoder")

    return decoder

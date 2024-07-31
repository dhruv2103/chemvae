import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model

from chemvae.decoder import decoder_model
from chemvae.encoder import encoder_model
from chemvae.property_predictor import load_property_predictor, property_predictor_model
from chemvae.tgru_k2_gpu import TerminalGRU
from chemvae.sampling import variational_layers


def load_models(params, is_training=True):
    def identity(x):
        return tf.identity(x)

    # def K_params with kl_loss_var
    kl_loss_var = tf.Variable(1, dtype=tf.float32)

    if params['reload_model']:
        encoder = load_encoder(params)
        decoder = load_decoder(params)
    else:
        encoder = encoder_model(params)
        decoder = decoder_model(params)

    x_in = encoder.inputs[0]

    z_mean, enc_output = encoder(x_in)
    z_samp, z_mean_log_var_output = variational_layers(z_mean, enc_output, kl_loss_var, params, is_training)

    # Decoder
    if params['do_tgru']:
        x_out = decoder([z_samp, x_in])
    else:
        x_out = decoder(z_samp)

    x_out = layers.Lambda(identity, name='x_pred', output_shape=x_out.shape)(x_out)
    model_outputs = [x_out, z_mean_log_var_output]

    AE_only_model = Model(x_in, model_outputs)

    if params['do_prop_pred']:
        if params['reload_model'] == True:
            property_predictor = load_property_predictor(params)
        else:
            property_predictor = property_predictor_model(params)

        if (('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0) and
                ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0)):

            reg_prop_pred, logit_prop_pred = property_predictor(z_mean)
            reg_prop_pred = layers.Lambda(identity, name='reg_prop_pred', output_shape=reg_prop_pred.shape)(
                reg_prop_pred)
            logit_prop_pred = layers.Lambda(identity, name='logit_prop_pred', output_shape=logit_prop_pred.shape)(
                logit_prop_pred)
            model_outputs.extend([reg_prop_pred, logit_prop_pred])

        # regression only scenario
        elif ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0):
            reg_prop_pred = property_predictor(z_mean)
            reg_prop_pred = layers.Lambda(identity, name='reg_prop_pred', output_shape=reg_prop_pred.shape)(
                reg_prop_pred)
            model_outputs.append(reg_prop_pred)

        # logit only scenario
        elif ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0):
            logit_prop_pred = property_predictor(z_mean)
            logit_prop_pred = layers.Lambda(identity, name='logit_prop_pred', output_shape=logit_prop_pred.shape)(
                logit_prop_pred)
            model_outputs.append(logit_prop_pred)

        else:
            raise ValueError('no logit tasks or regression tasks specified for property prediction')

        # making the models:
        AE_PP_model = Model(x_in, model_outputs)
        return AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var

    else:
        return AE_only_model, encoder, decoder, kl_loss_var


def load_encoder(params):
    encoder = encoder_model(params)
    encoder.load_weights(params['encoder_weights_file'])
    return encoder


def load_decoder(params):
    if params['do_tgru']:
        return load_model(params['decoder_weights_file'], custom_objects={'TerminalGRU': TerminalGRU})
    else:
        return load_model(params['decoder_weights_file'])


def kl_loss(truth_dummy, x_mean_log_var_output):
    x_mean, x_log_var = tf.split(x_mean_log_var_output, 2, axis=1)
    # print('x_mean shape in kl_loss: ', x_mean.get_shape())
    klloss = - 0.5 * tf.reduce_mean(1 + x_log_var - tf.square(x_mean) - tf.exp(x_log_var), axis=-1)
    return klloss

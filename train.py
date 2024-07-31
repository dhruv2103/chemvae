import os
import time
import argparse

import numpy as np
import tensorflow as tf
from functools import partial
from chemvae import hyperparameters
from chemvae.model import load_models, kl_loss
from chemvae.mol_callbacks import sigmoid_schedule, WeightAnnealer_epoch
from chemvae.preprocessing import vectorize_data
from tensorflow.keras.callbacks import CSVLogger


def main_property_run(params):
    start_time = time.time()

    # load data
    X_train, X_test, Y_train, Y_test = vectorize_data(params)

    print("X Train Shape: ", X_train.shape)
    print("X Test Shape: ", X_test.shape)

    print("Y Train Shape: ", Y_train[0].shape)
    print("Y Test Shape: ", Y_test[0].shape)


def main_no_prop(params):
    start_time = time.time()

    X_train, X_test = vectorize_data(params)

    print("X Train Shape: ", X_train.shape)
    print("X Test Shape: ", X_test.shape)

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()

    # Define the model within the strategy scope
    with strategy.scope():
        AE_only_model, encoder, decoder, kl_loss_var = load_models(params, is_training=True)

        # compile models
        if params['optim'] == 'adam':
            optim = tf.keras.optimizers.Adam(learning_rate=params['lr'], beta_1=params['momentum'])
        elif params['optim'] == 'rmsprop':
            optim = tf.keras.optimizers.RMSprop(learning_rate=params['lr'], rho=params['momentum'])
        elif params['optim'] == 'sgd':
            optim = tf.keras.optimizers.SGD(learning_rate=params['lr'], momentum=params['momentum'])
        else:
            raise NotImplemented("Please define valid optimizer")

        model_losses = {'x_pred': params['loss'],
                        'z_mean_log_var': kl_loss}

        # vae metrics, callbacks
        vae_sig_schedule = partial(sigmoid_schedule, slope=params['anneal_sigmod_slope'],
                                   start=params['vae_annealer_start'])
        vae_anneal_callback = WeightAnnealer_epoch(
            vae_sig_schedule, kl_loss_var, params['kl_loss_weight'], 'vae')

        csv_clb = CSVLogger(params["history_file"], append=False)
        callbacks = [vae_anneal_callback, csv_clb]

        def vae_anneal_metric(y_true, y_pred):
            return kl_loss_var

        xent_loss_weight = tf.Variable(params['xent_loss_weight'])

        # Distribute the dataset
        batch_size = params['batch_size']

        # Distribute Train Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, {
            'x_pred': X_train,
            'z_mean_log_var': np.ones((np.shape(X_train)[0], params['hidden_dim'] * 2))
        }))

        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        distributed_train_dataset = strategy.experimental_distribute_dataset(train_dataset)

        # Distribute Test Dataset
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, {
            'x_pred': X_test,
            'z_mean_log_var': np.ones((np.shape(X_test)[0], params['hidden_dim'] * 2))
        }))
        test_dataset = test_dataset.batch(batch_size)
        distributed_test_dataset = strategy.experimental_distribute_dataset(test_dataset)

        loss_wghts = {'x_pred': float(xent_loss_weight.numpy()),
                      'z_mean_log_var': float(kl_loss_var.numpy())}

        AE_only_model.compile(
            loss=model_losses,
            loss_weights=loss_wghts,
            optimizer=optim,
            metrics={'x_pred': ['categorical_accuracy', vae_anneal_metric]}
        )

        keras_verbose = params['verbose_print']

        AE_only_model.fit(distributed_train_dataset,
                          batch_size=params['batch_size'],
                          epochs=params['epochs'],
                          initial_epoch=params['prev_epochs'],
                          callbacks=callbacks,
                          verbose=keras_verbose,
                          validation_data=distributed_test_dataset
                          )

    encoder.save(params['encoder_weights_file'])
    decoder.save(params['decoder_weights_file'])
    print('time of run : ', time.time() - start_time)
    print('**FINISHED**')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_file',
                        help='experiment file', default='exp.json')
    parser.add_argument('-d', '--directory',
                        help='exp directory', default=None)
    args = vars(parser.parse_args())
    if args['directory'] is not None:
        args['exp_file'] = os.path.join(args['directory'], args['exp_file'])

    params = hyperparameters.load_params(args['exp_file'])
    print("All params:", params)

    # params['do_prop_pred'] = False

    if params['do_prop_pred']:
        main_property_run(params)
    else:
        main_no_prop(params)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model


def property_predictor_model(params):
    if ('reg_prop_tasks' not in params) and ('logit_prop_tasks' not in params):
        raise ValueError('You must specify either regression tasks and/or logistic tasks for property prediction')

    ls_in = layers.Input(shape=(params['hidden_dim'],),
                         name='prop_pred_input')

    prop_mid = layers.Dense(units=params['prop_hidden_dim'],
                            activation=params['prop_pred_activation'])(ls_in)

    if params['prop_pred_dropout'] > 0:
        prop_mid = layers.Dropout(rate=params['prop_pred_dropout'])(prop_mid)

    if params['prop_pred_depth'] > 1:
        for p_i in range(1, params['prop_pred_depth']):
            prop_mid = layers.Dense(units=int(params['prop_hidden_dim'] * params['prop_growth_factor'] ** p_i),
                                    activation=params['prop_pred_activation'],
                                    name="property_predictor_dense{}".format(p_i))(prop_mid)

            if params['prop_pred_dropout'] > 0:
                prop_mid = layers.Dropout(rate=params['prop_pred_dropout'])(prop_mid)

            if 'prop_batchnorm' in params and params['prop_batchnorm']:
                prop_mid = layers.BatchNormalization()(prop_mid)

    # for regression tasks
    if ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0):
        reg_prop_pred = layers.Dense(units=len(params['reg_prop_tasks']),
                                     activation='linear',
                                     name='reg_property_output')(prop_mid)

    # for logistic tasks
    if ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0):
        logit_prop_pred = layers.Dense(units=len(params['logit_prop_tasks']),
                                       activation='sigmoid',
                                       name='logit_property_output')(prop_mid)

    # both regression and logistic
    if (('reg_prop_tasks' in params) and
            (len(params['reg_prop_tasks']) > 0) and
            ('logit_prop_tasks' in params) and
            (len(params['logit_prop_tasks']) > 0)):

        property_predictor = Model(ls_in, [reg_prop_pred, logit_prop_pred], name="property_predictor")

    elif (('reg_prop_tasks' in params) and
          (len(params['reg_prop_tasks']) > 0)):

        property_predictor = Model(ls_in, reg_prop_pred, name="property_predictor")

    else:
        property_predictor = Model(ls_in, logit_prop_pred, name="property_predictor")

    return property_predictor


def load_property_predictor(params):
    return load_model(params['prop_pred_weights_file'])
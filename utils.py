"""
Useful methods to load the benchmark BCI IV2a dataset and to define EEGNet (using Keras)

EEGNet from https://doi.org/10.1088/1741-2552/aace8c.
BCI IV2a from https://doi.org/10.3389/fnins.2012.00055.

Author
------
Davide Borra, 2022
"""

from tensorflow import keras
import numpy as np
from scipy.io import loadmat


def load_bci_iv2a(fpath):
    '''This function accepts as input the filepath to a .mat containing signals from the BCI IV 2a dataset and
    returns the training and test data (tuples of x and y).
    Furthermore it returns also the sampling frequency, an ordered list containing the channel names (as contained in x),
    and a list of motor conditions.
    '''
    data = loadmat(fpath)
    #'x', 'y', 'channels', 'sf', 'tmin', 'tmax', 'fmin', 'fmax', 'events', 'session'
    x = data['x']
    y = np.squeeze(data['y']) - 1 # label encoding
    ch_names = [c[0] for c in np.squeeze(data['channels'])]
    srate = data['sf']
    #tmin, tmax = data['tmin'], data['tmax']
    #fmin, fmax = data['tmin'], data['tmax']
    session = np.squeeze(data['session'])
    conditions = [c[0] for c in np.squeeze(data['events'])]

    idx_train = np.where(session=='session_T')[0]
    idx_test = np.where(session=='session_E')[0]

    x_train = x[idx_train, :, :]
    x_test = x[idx_test, :, :]

    y_train = y[idx_train]
    y_test = y[idx_test]
    return (x_train, y_train), (x_test, y_test), srate, ch_names, conditions


def EEGNet(input_shape, # shape of the input EEG (channels, time, 1)
           n_classes, # number of output classes
           p_drop=0.5, # dropout probability (from 0 to 0.5) for hidden layers
           temporal_ks=(1, 65), # temporal kernel size (1st conv. layer)
           n_temporal_kernels=8, # number of temporal kernels (1st conv. layer)
           spatial_depth_multiplier=2, # spatial depthwise multiplier (2nd conv. layer)
           separable_temporal_ks=(1,17)): # separable temporal kernel size (3rd conv. layer)
    '''
    This function defines EEGNet as in Lawhern et al. (2018): 10.1088/1741-2552/aace8c.
    The hyper-parameters (including the default ones) are set as in  Borra et al. (2019): 10.1007/978-3-030-31635-8_223.
    The mandatory input parameters are the input EEG shape (channels, time, 1) and the number of classes to predict.
    '''

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),

            keras.layers.Conv2D(n_temporal_kernels,
                                kernel_size=temporal_ks,
                                padding='same',
                                use_bias = False),
            keras.layers.BatchNormalization(),
            keras.layers.DepthwiseConv2D(kernel_size=(input_shape[0], 1),
                                        use_bias=False,
                                        depth_multiplier=spatial_depth_multiplier,
                                        depthwise_constraint=keras.constraints.max_norm(1.)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('elu'),

            keras.layers.AveragePooling2D(pool_size=(1, 4)),
            keras.layers.Dropout(p_drop),

            keras.layers.SeparableConv2D(n_temporal_kernels*spatial_depth_multiplier,
                                         kernel_size=separable_temporal_ks,
                                         use_bias=False,
                                         padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('elu'),

            keras.layers.AveragePooling2D(pool_size=(1, 8)),
            keras.layers.Dropout(p_drop),

            keras.layers.Flatten(),

            keras.layers.Dense(n_classes,
                               kernel_constraint=keras.constraints.max_norm(0.25),
                               activation="softmax"),
        ]
    )
    return model

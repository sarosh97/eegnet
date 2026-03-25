#!/usr/bin/env python
# coding: utf-8

"""
Script for training a convolutional neural network to classify imagined motor conditions from single EEG trials.
The convolutional neural network is EEGNet and the dataset is from the BCI IV2a benchmark dataset (the needed functionalities are provided in the utils.py script).

EEGNet from https://doi.org/10.1088/1741-2552/aace8c.
BCI IV2a from https://doi.org/10.3389/fnins.2012.00055.

Author
------
Sarosh Ali Shah, 2023
"""

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from utils import load_bci_iv2a, EEGNet


# # Exercise 11 – Classification of motor imagery with a convolutional neural network from single EEG trials

# In[ ]:


def to_one_hot(y_dense, C):
    '''Function that transform labels from label encoding (e.g., l \in [0,...,3])
    to one-hot encoding.
    Example: l_i=2 (label enc.) -> l_i=[0, 0, 1, 0] (one-hot enc.)
    '''
    n_examples = y_dense.shape[0]
    y = np.zeros((n_examples, C), dtype=int)
    for i in np.arange(n_examples):
        y[i, y_dense[i]] = 1
    return y


# In[ ]:


# data pre-processing (if needed) and split
n_chans = 22 # number of channels
n_time = 256 # number of time steps
C = 4 # number of total classes

# load the data and split it between train and test sets
(x_train, labels_train), (x_test, labels_test), srate, ch_names, conditions = \
    load_bci_iv2a('bci_iv2a_sub-008.mat')

print("Input data type:", x_train.dtype)
print("Shape of the examples (training set):", x_train.shape)
print("Shape of the examples (test set):", x_test.shape)

print("Min value (training examples):", x_train.min())
print("Max value (training examples):", x_train.max())

# make sure images have shape (channels, time, 1)
n_examples_train = x_train.shape[0]
n_examples_test = x_test.shape[0]
x_train = x_train.reshape((n_examples_train, n_chans, n_time, 1))
x_test = x_test.reshape((n_examples_test, n_chans, n_time, 1))
print("Shape of the examples (training set) after reshaping:", x_train.shape)
print("Shape of the examples (test set) after reshaping:", x_test.shape)

# convert labels to one-hot encoded labels using a custom method
print("Shape of the training labels:", labels_train.shape)
print("Shape of the test labels:", labels_test.shape)
y_train = to_one_hot(labels_train, C)
y_test = to_one_hot(labels_test, C)
print("Shape of the training labels (after one-hot encoding):", y_train.shape)
print("Shape of the test labels (after one-hot encoding):", y_test.shape)

# extracting the validation set as the first 10% of examples
valid_ratio = 0.1 # ratio of the overall training set to held back as validation set
x_valid = x_train[:round(valid_ratio*n_examples_train),:,:,:]
y_valid = y_train[:round(valid_ratio*n_examples_train),:]
labels_valid = labels_train[:round(valid_ratio*n_examples_train)]
# assigning back the training set as the remaining 90% of examples
x_train = x_train[round(valid_ratio*n_examples_train):,:,:,:]
y_train = y_train[round(valid_ratio*n_examples_train):,:]
labels_train = labels_train[round(valid_ratio*n_examples_train):]

print("Shape of the examples (training set):", x_train.shape)
print("Shape of the examples (validation set):", x_valid.shape)
print("Shape of the examples (test set):", x_test.shape)

# standardize x using statistics (mean and standard deviation) computed on the training set
m = np.mean(x_train)
s = np.std(x_train)
x_train = (x_train-m)/(1e-15+s)
x_valid = (x_valid-m)/(1e-15+s)
x_test = (x_test-m)/(1e-15+s)
print("Min value (training examples) after standardization:", x_train.min())
print("Max value (training examples) after standardization:", x_train.max())


# In[ ]:


# histogram of classes per set (training, validation, test)
plt.figure(figsize=(11,8))
plt.subplot(1,3,1)
n_examples_perclass = []
for c in np.unique(labels_train):
    n_examples_perclass.append(np.sum(labels_train==c))
plt.bar(x=np.arange(len(conditions)), height=n_examples_perclass, facecolor='grey', edgecolor='k')
plt.ylim([0, 80])
plt.xticks(np.arange(len(conditions)),
           conditions,
           rotation=30)
plt.title('Training set')
plt.ylabel('no. of examples')
plt.xlabel('class')

plt.subplot(1,3,2)
n_examples_perclass = []
for c in np.unique(labels_valid):
    n_examples_perclass.append(np.sum(labels_valid==c))
plt.bar(x=np.arange(len(conditions)), height=n_examples_perclass, facecolor='grey', edgecolor='k')
plt.ylim([0, 80])
plt.xticks(np.arange(len(conditions)),
           conditions,
           rotation=30)
plt.title('Validation set')
plt.ylabel('no. of examples')
plt.xlabel('class')

plt.subplot(1,3,3)
n_examples_perclass = []
for c in np.unique(labels_test):
    n_examples_perclass.append(np.sum(labels_test==c))
plt.bar(x=np.arange(len(conditions)), height=n_examples_perclass, facecolor='grey', edgecolor='k')
plt.ylim([0, 80])
plt.xticks(np.arange(len(conditions)),
           conditions,
           rotation=30)
plt.title('Test set')
plt.ylabel('no. of examples')
plt.xlabel('class')

plt.tight_layout()
plt.show()


# In[ ]:


# network design
model = EEGNet((n_chans, n_time, 1), C) # model initialization
model.summary() # model details


# In[ ]:


# network training
lr = 0.001 # learning rate
momentum = 0.9 # momentum term
# defining the optimizer (stochastic gradient descent with momentum)
optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)

# compiling the model=defining the desired loss function to be minimized, the algorithm to use for the optimization process, and other metrics to track
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# defining the ModelCheckpoint callback
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='best_mdl.keras', # set to 'best_mdl.keras' if you want only the best model, overall or to '{epoch:02d}-{val_loss:.5f}.keras' if you want to save the best model as the training proceed (best model over time)
    monitor='val_loss', # set to the metric that you want to track for the early stopped model (evaluated offline)
    save_best_only=True)

# start optimizing the network
batch_size = 32 # mini-batch size
max_epochs = 1000 # maximum number of epochs
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=max_epochs,
                    validation_data=(x_valid, y_valid),
                    callbacks=[model_checkpoint_callback])

# extracting training and validation losses, training and validation accuracies
train_loss = history.history['loss']
train_acc = history.history['accuracy']
valid_loss = history.history['val_loss']
valid_acc = history.history['val_accuracy']


# In[ ]:


# plot losses and accuracies
epochs = np.arange(1, len(train_loss)+1)

plt.figure(figsize=(11,8))
plt.subplot(2,1,1)
plt.plot(epochs, train_loss, 'k')
plt.plot(epochs, valid_loss, 'r')
plt.ylabel('loss')
plt.xlabel('epochs')

plt.subplot(2,1,2)
plt.plot(epochs, train_acc, 'k')
plt.plot(epochs, valid_acc, 'r')
plt.ylabel('accuracy')
plt.xlabel('epochs')

plt.legend(['training', 'validation'])
plt.show()


# In[ ]:


# model evaluation (on training, validation and test sets)

# loading best model
model = keras.models.load_model('best_mdl.keras')

proba = model.predict(x_train) # g(X_test; theta_trained_best_model)
y_pred = np.argmax(proba, axis=-1)
cmtx = confusion_matrix(y_true=labels_train, y_pred=y_pred)
print("#"*10+'Training set')
print('Confusion matrix')
print(cmtx)
print("Accuracy: ", np.mean(labels_train==y_pred))

proba = model.predict(x_valid)
y_pred = np.argmax(proba, axis=-1)
cmtx = confusion_matrix(y_true=labels_valid, y_pred=y_pred)
print("#"*10+'Validation set')
print('Confusion matrix')
print(cmtx)
print("Accuracy: ", np.mean(labels_valid==y_pred))

proba = model.predict(x_test)
y_pred = np.argmax(proba, axis=-1)
cmtx = confusion_matrix(y_true=labels_test, y_pred=y_pred)
print("#"*10+'Test set')
print('Confusion matrix')
print(cmtx)
print("Accuracy: ", np.mean(labels_test==y_pred))


# In[ ]:


# visualization of spatial weights or aggregation across spatial weights (e.g., max or mean of absolute values)
layers = model.layers # list containing all trained layers
print(layers)
layer = layers[2] # spatial conv. layer
weights = layer.get_weights()[0] # accessing weights
print(layer.name, weights.shape)

weights = weights.reshape((weights.shape[0], -1)) # reshaping (can be avoided)
# weights now is (22, 16)
nspat_filters = weights.shape[1]
weights = np.abs(weights) # abs value
weights = np.mean(weights, axis=1) # averaging

plt.figure(figsize=(11,8))
x = np.arange(weights.shape[0])
y = weights
plt.bar(x=x,
        height=y,
        facecolor='grey',
        edgecolor='k')
plt.ylabel('average absolute weight')
plt.xlabel('EEG channel')
plt.xticks(ticks=x, labels=ch_names)
plt.show()

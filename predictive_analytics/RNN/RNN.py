

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, metrics
import numpy as np
from functions import evaluate_on_ticker, evaluate, get_last_step_predictions, get_last_step_accuracy_based_on_confidence # custom-made helper functions


INTERVAL = 5

X_train = np.load('../../data/transformed/{}min/train_data.npy'.format(INTERVAL))
y_train = np.load('../../data/transformed/{}min/train_targets.npy'.format(INTERVAL))
X_test = np.load('../../data/transformed/{}min/test_data.npy'.format(INTERVAL))
y_test = np.load('../../data/transformed/{}min/test_targets.npy'.format(INTERVAL))



# Create model 
num_neurons = 30
model = models.Sequential([
    layers.LSTM(num_neurons, return_sequences=True, input_shape=[None, 5]),
    layers.LSTM(num_neurons, return_sequences=True),
    layers.LSTM(num_neurons, return_sequences=True),
    layers.LSTM(10, return_sequences=True),
    layers.TimeDistributed(layers.Dense(3, activation='softmax'))
    ])

def last_step_accuracy(Y_true, Y_pred):
    last_step_labels = tf.dtypes.cast(Y_true[:, -1], tf.int32)
    last_step_preds = Y_pred[:,-1]
    last_step_preds = tf.math.argmax(last_step_preds, axis=1, output_type=tf.int32)
    compare = tf.dtypes.cast(tf.equal(last_step_preds, last_step_labels), tf.int32)
    tot_correct = tf.reduce_sum(compare)
    tot_size = tf.size(last_step_labels)
    accuracy = tf.divide(tot_correct, tot_size)
    return accuracy

optimizer = optimizers.Adam(lr=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[last_step_accuracy])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15)

    
evaluate(model, X_test, y_test)
get_last_step_accuracy_based_on_confidence(model, X_test, y_test, 0.90)






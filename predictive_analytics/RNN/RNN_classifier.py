

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, metrics
import numpy as np
from functions import evaluate_on_ticker, get_last_step_predictions, get_last_step_performance_based_on_confidence # custom-made helper functions
from sklearn.utils import class_weight
import os


SAVE = False
MODEL_NAME = 'beta_1'

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
    layers.LSTM(30, return_sequences=True),
    layers.TimeDistributed(layers.Dense(3, activation='softmax'))
    ])

@tf.autograph.experimental.do_not_convert
def last_step_accuracy(Y_true, Y_pred):
    last_step_labels = tf.dtypes.cast(Y_true[:, -1], tf.int32)
    last_step_preds = Y_pred[:,-1]
    last_step_preds = tf.math.argmax(last_step_preds, axis=1, output_type=tf.int32)
    compare = tf.dtypes.cast(tf.equal(last_step_preds, last_step_labels), tf.int32)
    tot_correct = tf.reduce_sum(compare)
    tot_size = tf.size(last_step_labels)
    accuracy = tf.divide(tot_correct, tot_size)
    return accuracy

PATH_TO_MODEL = "models/{}".format(MODEL_NAME)
if os.path.isdir(PATH_TO_MODEL):
    model = tf.keras.models.load_model(PATH_TO_MODEL, custom_objects={'last_step_accuracy': last_step_accuracy})

else:
    class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train[:,-1])
    class_weights_dict = {i: class_weights[i] for i in [0,1,2]}
    sample_weights = class_weight.compute_sample_weight(class_weights_dict, y_train[:,-1])
    
    optimizer = optimizers.Adam(lr=0.01)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='last_step_accuracy', mode='max', patience=3)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[last_step_accuracy])
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=8, 
                        sample_weight=sample_weights, callbacks=[early_stopping],
                        batch_size=32)
    if SAVE:
        model.save(PATH_TO_MODEL) 

# Evaluate
get_last_step_performance_based_on_confidence(model, X_test, y_test, 0)
get_last_step_performance_based_on_confidence(model, X_test, y_test, 0.9)


# Test on new data
ticker = 'QCOM'
START_DATE = '2022-10-22'
END_DATE = '2022-11-20'
result = evaluate_on_ticker(model, ticker, START_DATE, END_DATE, conf_thresholds=[0,0.7,0.9,0.95])

# repeated nested cross validation

# 




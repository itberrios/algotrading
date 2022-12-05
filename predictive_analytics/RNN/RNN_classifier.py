

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, metrics
import numpy as np
from functions import evaluate_on_ticker, get_last_step_predictions, get_last_step_performance_based_on_confidence # custom-made helper functions
from sklearn.utils import class_weight
import os
                

# =============================================================================
# Parameters & Data
# =============================================================================
    
MODEL_NAME = 'beta_1'
SAVE = False                # save model weights after training
USE_EXISTING = False        # use existing model with MODEL_NAME if present
CROSS_VALIDATION = True     # run nested cross validation if true, else run once
INTERVAL = 5                # use INTERVAL-minute stock data

X_train = np.load('../../data/transformed/{}min/train_data.npy'.format(INTERVAL))
y_train = np.load('../../data/transformed/{}min/train_targets.npy'.format(INTERVAL))
X_test = np.load('../../data/transformed/{}min/test_data.npy'.format(INTERVAL))
y_test = np.load('../../data/transformed/{}min/test_targets.npy'.format(INTERVAL))


# =============================================================================
# Custom functions & metrics
# =============================================================================

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

def nested_cross_validation(X, y, k, 
                            param_grid={'num_layers':[2,3], 'num_neurons':[10,15,30], 'learning_rate':[0.01, 0.02]}
                            ):
    
    fold_size = X.shape[0] // k
    result = []
    for num_layers in param_grid['num_layers']:
        for neuron_per_layer in param_grid['num_neurons']:
            for learning_rate in param_grid['learning_rate']:
                params_str = '[num_layer={}, num_neuron={}, lrate={}]'.format(num_layers, neuron_per_layer, learning_rate)
                print("Working with", params_str)
                
                # nested k-fold
                fold_result = []
                val_start_idx = fold_size
                for fold in range(k-1):
                    # reset model
                    input_layer = [layers.LSTM(neuron_per_layer, return_sequences=True, input_shape=[None,5])]
                    middle_layers = [layers.LSTM(neuron_per_layer, return_sequences=True) for _ in range(num_layers-1)]
                    output_layer = [layers.TimeDistributed(layers.Dense(3, activation='softmax'))]
                    model = models.Sequential(input_layer + middle_layers + output_layer)
                    
                    # provision training and validating sets
                    val_end_idx = val_start_idx + fold_size
                    training_X, training_y = X[:val_start_idx], y[:val_start_idx]
                    val_X, val_y = X[val_start_idx:val_end_idx], y[val_start_idx:val_end_idx]
                    
                    # sample weights based on label distribution
                    class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(training_y), y=training_y[:,-1])
                    class_weights_dict = {i: class_weights[i] for i in [0,1,2]}
                    sample_weights = class_weight.compute_sample_weight(class_weights_dict, training_y[:,-1])
                    
                    # cost function & callbacks
                    optimizer = optimizers.Adam(lr=learning_rate)
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='last_step_accuracy', mode='max', patience=2)
                    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[last_step_accuracy])
                    
                    # training start
                    history = model.fit(training_X, training_y, validation_data=(val_X, val_y), epochs=8, 
                                        sample_weight=sample_weights, callbacks=[early_stopping],
                                        batch_size=32, verbose=1)
                    fold_result.append(history.history['val_last_step_accuracy'][-1])
                    
                # aggregate & save folds' result
                avg_val_metric = np.mean(fold_result) 
                print(params_str, avg_val_metric)
                result.append((params_str, avg_val_metric))
    
    # sort results
    result.sort(key=lambda x: x[1], reverse=True) # sorted by descending validation metric
    return result
                    

# =============================================================================
# Training & Testing
# =============================================================================

PATH_TO_MODEL = "models/{}".format(MODEL_NAME)
if USE_EXISTING and os.path.isdir(PATH_TO_MODEL):
    model = tf.keras.models.load_model(PATH_TO_MODEL, custom_objects={'last_step_accuracy': last_step_accuracy})
    
elif not CROSS_VALIDATION:
    # Create model 
    num_neurons = 30
    model = models.Sequential([
        layers.LSTM(num_neurons, return_sequences=True, input_shape=[None, 5]),
        layers.LSTM(num_neurons, return_sequences=True),
        layers.LSTM(num_neurons, return_sequences=True),
        layers.LSTM(30, return_sequences=True),
        layers.TimeDistributed(layers.Dense(3, activation='softmax'))
        ])
    
    class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train[:,-1])
    class_weights_dict = {i: class_weights[i] for i in [0,1,2]}
    sample_weights = class_weight.compute_sample_weight(class_weights_dict, y_train[:,-1])
    
    optimizer = optimizers.Adam(lr=0.01)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='last_step_accuracy', mode='max', patience=3)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[last_step_accuracy])
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, 
                        sample_weight=sample_weights, callbacks=[early_stopping],
                        batch_size=32)
    if SAVE:
        model.save(PATH_TO_MODEL) 
        
else:       # Nested cross-validation
    param_grid = {'num_layers':[2,3], 'num_neurons':[10,15,30], 'learning_rate':[0.01, 0.02]};
    print("Nested cross validation starting with param_grid =", param_grid)
    result = nested_cross_validation(X_train, y_train, 5, param_grid)
    print("Nested cross validation results:")
    for r in result:
        print('\t', r)
        



                    

# # Evaluate
# get_last_step_performance_based_on_confidence(model, X_test, y_test, 0)
# get_last_step_performance_based_on_confidence(model, X_test, y_test, 0.9)


# # Test on new data
# ticker = 'QCOM'
# START_DATE = '2022-10-22'
# END_DATE = '2022-11-20'
# result = evaluate_on_ticker(model, ticker, START_DATE, END_DATE, conf_thresholds=[0,0.7,0.9,0.95])





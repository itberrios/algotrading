import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class WindowGenerator():
    def __init__(self, input_width, label_width, shift, dfs,
                batch_size=32, seed=42, window_norm=True,
                sample_weights=True, 
                remove_columns=[],
                label_columns=None,
                remove_nonsequential=False):
        # Store the raw data.
        self.train_df = dfs[0]
        self.valid_df = dfs[1]
        self.test_df = dfs[2]
      
        # self.position_encode = position_encode
        self.batch_size = batch_size
        self.seed = seed
        self.window_norm = window_norm
        self.sample_weights = sample_weights
        self.remove_columns = remove_columns
        self.label_columns = label_columns
        self.remove_nonsequential = remove_nonsequential # removes non-sequential windows

        # standardize training features if window norm not selected
        if not self.window_norm:
            self.standardize()

        # Work out the label column indices.
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                                enumerate(self.train_df.columns)} 

        # Work out the window parameters.
        self.input_width = input_width # sequence length
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]


    def __repr__(self):
      return '\n'.join([
          f'Total window size: {self.total_window_size}',
          f'Input indices: {self.input_indices}',
          f'Label indices: {self.label_indices}',
          f'Label column name(s): {self.label_columns}'])

    def standardize(self):
        train_mean = self.train_df.mean()
        train_std = self.train_df.std()

        # ensure that target column is not standardized
        for col in set(self.label_columns + self.remove_columns):
            train_mean[col] = 0
            train_std[col] = 1

        self.train_df = (self.train_df - train_mean) / train_std
        self.valid_df = (self.valid_df - train_mean) / train_std
        self.test_df = (self.test_df - train_mean) / train_std


    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        if self.remove_nonsequential:
            inputs, labels = self.remove_nonsequential_windows(inputs, labels)

        # remove desired columns from input features 
        if len(self.remove_columns) > 0:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]] for name in self.column_indices.keys() 
                                if name not in self.remove_columns],
                axis=-1)

        return inputs, labels


    def standardize_window(self, inputs, labels):
        ''' Standardizes each window to mean - 0 and std - 1'''
        mean = tf.math.reduce_mean(inputs, axis=1)
        std = tf.math.reduce_std(inputs, axis=1)
        
        mean = tf.repeat(tf.expand_dims(mean, axis=1), 
                         self.total_window_size, axis=1)
        std = tf.repeat(tf.expand_dims(std, axis=1), 
                        self.total_window_size, axis=1)

        inputs = tf.math.subtract(inputs, mean)
        inputs = tf.math.divide(inputs, std)

        return inputs, labels


    def get_sample_weights(self, inputs, labels):
        ''' Obtains smaple weights for any number of classes.
            NOTE: sample_weights pertain a weighting to each label
            '''
        # get initial sample weights
        sample_weights = tf.ones_like(labels, dtype=tf.float64)
        
        # get classes and counts for each one
        class_counts = np.bincount(self.train_df.price_change)
        total = class_counts.sum()
        n_classes = len(class_counts)

        for idx, count in enumerate(class_counts):
            # compute weight
            weight = total / (n_classes*count)

            # update weight value 
            sample_weights = tf.where(tf.equal(labels, float(idx)), 
                                      weight, 
                                      sample_weights)
        
        return inputs, labels, sample_weights


    def remove_sequence(self, inputs, labels, sample_weights=None):
        # remove sequence from inputs so simple models can be trained (i.e. Linear Models)
        inputs = tf.expand_dims(inputs[:, 0, :], axis=1)

        if tf.is_tensor(sample_weights):
            return inputs, labels, sample_weights
        else:
            return inputs, labels


    def remove_nonsequential_windows(self, inputs, labels, sample_weights=None):
        # get locations of consistent dayofweeks in each batch
        dayofweek_repeats = tf.repeat(tf.expand_dims(inputs[:, 0, -1], axis=1), 
                                      self.input_width, axis=1)
        valid_locs = tf.reduce_all(inputs[:, :, -1] == dayofweek_repeats, axis=1)

        inputs = inputs[valid_locs]
        labels = labels[valid_locs]

        if tf.is_tensor(sample_weights):
            return inputs, labels, sample_weights
        else:
            return inputs, labels
            
    
    def get_position_encoding(self, n=10000):
        d = self.train_df.shape[1] # assume all features are used
        pos_encode = np.zeros((self.input_width, d))
        for k in range(self.input_width):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                pos_encode[k, 2*i] = np.sin(k/denominator)
                pos_encode[k, 2*i+1] = np.cos(k/denominator)
        return pos_encode

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=False,
                seed=self.seed,
                batch_size=self.batch_size)

        # get split window
        ds = ds.map(self.split_window)

        if self.window_norm:
            ds = ds.map(self.standardize_window)

        if self.sample_weights:
            ds = ds.map(self.get_sample_weights)

        return ds
    

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def valid(self):
        return self.make_dataset(self.valid_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        # return result
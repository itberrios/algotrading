import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class WindowGenerator():
    def __init__(self, input_width, label_width, shift, dfs,
                batch_size=32, shuffle=False, seed=42,
                sample_weights=True,
                remove_labels_from_inputs=False, # update to remove any column from inputs
                label_columns=None):
      # Store the raw data.
      self.train_df = dfs[0]
      self.valid_df = dfs[1]
      self.test_df = dfs[2]

      # self.train_mean = train_mean
      # self.train_std = train_std
      
      # self.position_encode = position_encode
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.seed = seed
      self.sample_weights = sample_weights
      self.remove_labels_from_inputs = remove_labels_from_inputs

      # Work out the label column indices.
      self.label_columns = label_columns
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


    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # remove label from input features 
        if self.remove_labels_from_inputs:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]] for name in self.column_indices.keys() 
                                if name not in self.label_columns],
                axis=-1)

        
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels


    def normalize(self, inputs, labels):
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
        # weights = tf.ones(shape=(32, 1))*0.33 # compute_sample_weight(class_weight='balanced', y=labels)
        # weights[labels == 0] *= 1.5
        # weights[labels == 2] *= 1.5

        # compute sample weights
        # num_down, num_same, num_up = np.bincount(self.train_df.price_change)

        # get initial sample weights
        sample_weights = tf.ones_like(labels, dtype=tf.float64)
        
        # get classes and counts for each one
        class_counts = np.bincount(self.train_df.price_change)
        total = class_counts.sum()
        n_classes = len(class_counts)

        weights = []
        for idx, count in enumerate(class_counts):
            # compute weight
            weight = total / (n_classes*count)

            # update weight value 
            sample_weights = tf.where(tf.equal(labels, float(idx)), 
                                      weight, 
                                      sample_weights)
        
        return inputs, labels, sample_weights


    def plot(self, data, model=None, plot_col='price_diff', max_subplots=3):
        inputs, labels = data
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
          plt.subplot(max_n, 1, n+1)
          plt.ylabel(f'{plot_col} [normed]')
          plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                  label='Inputs', marker='.', zorder=-10)

          if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
          else:
            label_col_index = plot_col_index

          if label_col_index is None:
            continue

          plt.scatter(self.label_indices, labels[n, :, label_col_index],
                      edgecolors='k', label='Labels', c='#2ca02c', s=64)
          if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

          if n == 0:
            plt.legend()

        plt.xlabel('Time [h]')

    # WindowGenerator.plot = plot
    
    
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
                shuffle=self.shuffle,
                seed=self.seed,
                batch_size=self.batch_size)

        ds = ds.map(self.split_window)
        ds = ds.map(self.normalize)

        if self.sample_weights:
            ds = ds.map(self.get_sample_weights)

        return ds

    # WindowGenerator.make_dataset = make_dataset
    

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
        return result

    # WindowGenerator.train = train
    # WindowGenerator.valid = valid
    # WindowGenerator.test = test
    # WindowGenerator.example = example
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                train_df, valid_df, test_df,
                # train_mean=None, train_std=None,
               batch_size=32, shuffle=False, seed=42,
                remove_labels_from_inputs=False, # update to remove any column from inputs
                label_columns=None):
      # Store the raw data.
      self.train_df = train_df
      self.valid_df = valid_df
      self.test_df = test_df

      # self.train_mean = train_mean
      # self.train_std = train_std
      
      # self.position_encode = position_encode
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.seed = seed
      self.remove_labels_from_inputs = remove_labels_from_inputs

      # Work out the label column indices.
      self.label_columns = label_columns
      if label_columns is not None:
         self.label_columns_indices = {name: i for i, name in
                                       enumerate(label_columns)}
      self.column_indices = {name: i for i, name in
                             enumerate(train_df.columns)} # 

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

        # add augmented data here
        # inputs[:, 3, :]

        # we need to resample with a roughly even amount of classes

        # add position encoding to the data
        # may need to concatenate this data?
        # if self.position_encode:
        #     pos_encode = self.get_position_encoding()
        #     pos_encode = np.repeat(pos_encode[None, :, :], self.batch_size, axis=0)

        #     inputs = inputs + tf.convert_to_tensor(pos_encode)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    # WindowGenerator.split_window = split_window


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
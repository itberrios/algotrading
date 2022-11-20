'''
   contains the basic Transformer Encoder Architecture
   usage:
   
   from basic_transformer import TransformerModel

   transformer_model = TransformerModel(
            n_heads=2,
            d_model=512,
            ff_dim=256,
            num_transformer_blocks=2,
            mlp_units=[256],
            n_outputs=3,
            dropout=0.1,
            mlp_dropout=0.1)


   Position Encoding Code from: https://www.tensorflow.org/text/tutorials/transformer.
'''


import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow_addons.layers import MultiHeadAttention



# ============================================================================
# Positional Encoding
def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(layers.Layer):
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.d_model = d_model
        self.ff_dim = ff_dim

    def build(self, input_shape):
        # self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
        self.embedding = layers.Dense(self.d_model)
        self.pos_encoding = positional_encoding(length=self.ff_dim, depth=self.d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

# ============================================================================
# Transformer Encoder

class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, n_heads, d_model, ff_dim, dropout=0):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.attn_heads = list()


    def build(self, input_shape):
        
        # attention portion
        self.attn_multi = MultiHeadAttention(num_heads=self.n_heads, 
                                             head_size=self.d_model, 
                                             dropout=self.dropout)
        self.attn_dropout = layers.Dropout(self.dropout)
        self.attn_norm = layers.LayerNormalization(epsilon=1e-6)

        # feedforward portion
        self.ff_conv1 = layers.Conv1D(filters=self.ff_dim, 
                                      kernel_size=1, 
                                      activation='relu')
        self.ff_dropout = layers.Dropout(self.dropout)
        self.ff_conv2 = layers.Conv1D(filters=input_shape[-1],
                                      kernel_size=1)
        self.ff_norm = layers.LayerNormalization(epsilon=1e-6)


    def call(self, inputs):
        # attention portion
        x = self.attn_multi([inputs, inputs])
        x = self.attn_dropout(x)
        x = self.attn_norm(x)

        # get first residual
        res = x + inputs
        
        # feedforward portion
        x = self.ff_conv1(res)
        x = self.ff_dropout(x)
        x = self.ff_conv2(x)
        x = self.ff_norm(x)
        
        # return residual
        return res + x

# ============================================================================
# Transformer Model main

class TransformerModel(keras.Model):

    def __init__(self, 
            n_heads,
            d_model,
            # d_head, # set same value for d_k and d_v
            ff_dim,
            num_transformer_blocks,
            mlp_units,
            n_outputs=3,
            dropout=0.1,
            mlp_dropout=0.1):
            
        super().__init__()
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.n_outputs = n_outputs
        self.dropout = dropout
        self.mlp_dropout = mlp_dropout
        
         
    def build(self, input_shape):

        # get embedding layer that projects inputs inot high dimensional space
        # self.embed = layers.Dense(self.d_model)

        # get learnable time layer
        # self.time_layer = layers.Layer(tf.random.uniform((input_shape[1], self.d_model), -0.2, 0.2))
        # self.time_layer = tf.Variable(
        #     initial_value=tf.random.uniform((input_shape[1], self.d_model), -0.2, 0.2)
        #     )
        
        # get positional embedding
        self.positional_embedding = PositionalEmbedding(self.d_model, self.ff_dim)

        # get transformer encoders
        self.encoders = [TransformerEncoder(self.n_heads, self.d_model, self.ff_dim, self.dropout) 
                         for _ in range(self.num_transformer_blocks)]

        self.avg_pool = layers.GlobalAveragePooling1D(data_format='channels_last') # batch, steps, features "channels_first")

        # get MLP portion of network
        self.mlp_layers = []
        for dim in self.mlp_units:
            self.mlp_layers.append(layers.Dense(dim, activation="relu"))
            self.mlp_layers.append(layers.Dropout(self.mlp_dropout))

        # output layer 
        self.mlp_output = layers.Dense(self.n_outputs, activation='softmax')


    def call(self, x):

        # project input data into high dimensional space
        # x = self.embed(x)

        # inject time information ??
        # x = x + self.time_layer(x)

        # TEMP
        # print(x.shape)

        # Project Input to high Dimensional Space and Encode Position Information
        x = self.positional_embedding(x)

        # TEMP
        # print(x.shape)
        
        # Encoder Portion
        for encoder in self.encoders:
            x = encoder(x)

        # TEMP
        # print(x.shape)

        # Average Pooling
        x = self.avg_pool(x)

        # TEMP
        # print('avg pool', x.shape)

        # MLP portion for classification
        for mlp_layer in self.mlp_layers:
            x = mlp_layer(x)

        # TEMP
        # print(x.shape)

        x = self.mlp_output(x)

        # TEMP
        # print(x.shape)

        return x

    # Needed for saving and loading model with custom layer
    # see: https://www.tensorflow.org/guide/keras/save_and_serialize
    def get_config(self): 
        config = super().get_config().copy()
        config.update({
                'n_heads' : self.n_heads,
                'd_model' : self.d_model,
                'ff_dim' : self.ff_dim,
                'num_transformer_blocks' : self.num_transformer_blocks,
                'mlp_units' : self.mlp_units,
                'n_outputs' : self.n_outputs,
                'dropout' : self.dropout,
                'mlp_dropout' : self.mlp_dropout})
        return config    

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
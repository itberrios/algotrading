from symbol import tfpdef


import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow_addons.layers import MultiHeadAttention


# ============================================================================
# Transformer Encoder

class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, n_heads, head_size, ff_dim, dropout=0):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_size = head_size
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.attn_heads = list()


    def build(self, input_shape):
        
        # attention portion
        self.attn_multi = MultiHeadAttention(num_heads=self.n_heads, 
                                             head_size=self.head_size, 
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
    
    # Needed for saving and loading model with custom layer
    def get_config(self): 
        config = super().get_config().copy()
        config.update({'d_k': self.d_k,
                       'd_v': self.d_v,
                       'n_heads': self.n_heads,
                       'ff_dim': self.ff_dim,
                       'attn_heads': self.attn_heads,
                       'dropout': self.dropout_rate})
        return config   

# ============================================================================
# Transformer Model main

class TransformerModel(keras.Model):

    def __init__(self, 
            n_heads,
            head_size,
            ff_dim,
            num_transformer_blocks,
            mlp_units,
            n_outputs=3,
            dropout=0.1,
            mlp_dropout=0.1):
            
        super().__init__()
        
        self.n_heads = n_heads
        self.head_size = head_size
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.n_outputs = n_outputs
        self.dropout = dropout
        self.mlp_dropout = mlp_dropout

        
         
    def build(self, input_shape):
        # get embedding layer that projects inputs inot high dimensional space
        self.embed = layers.Dense(self.head_size)
        
        # get transformer encoders
        self.encoders = [TransformerEncoder(self.n_heads, self.head_size, self.ff_dim, self.dropout) 
                         for _ in range(self.num_transformer_blocks)]

        self.avg_pool = layers.GlobalAveragePooling1D(data_format="channels_first")

        # get MLP portion of network
        self.mlp_layers = []
        for dim in self.mlp_units:
            self.mlp_layers.append(layers.Dense(dim, activation="relu"))
            self.mlp_layers.append(layers.Dropout(self.mlp_dropout))

        # output layer 
        self.mlp_output = layers.Dense(self.n_outputs, activation='softmax')


    def call(self, x):

        # project input data into high dimensional space
        x = self.embed(x)
        
        # Encoder Portion
        for encoder in self.encoders:
            x = encoder(x)

        # Average Pooling
        x = self.avg_pool(x)

        # MLP portion for classification
        for mlp_layer in self.mlp_layers:
            x = mlp_layer(x)

        x = self.mlp_output(x)

        return x

# ============================================================================
# Option to use Function for Transformer Model instead




def build_transformer(input_shape,
            n_heads,
            head_size,
            ff_dim,
            num_transformer_blocks,
            mlp_units,
            n_outputs=3,
            dropout=0.1,
            mlp_dropout=0.1,
        ):
    ''' Option to build transformer model 
        usage:
            input_shape = inputs.shape[1:]

            transformer_model = build_transformer(
                input_shape,
                n_heads=2,
                head_size=512,
                ff_dim=256,
                num_transformer_blocks=2,
                mlp_units=[256],
                n_outputs=3,
                dropout=0.1,
                mlp_dropout=0.1,
            )
        '''
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # model to project inputs into higher dimensional space (same as head_size??)
    # This is just a linear layer with a bias and no activation
    x = layers.Dense(units=head_size)(x)

    # model to encode time/positions into high dimensional data

    # encoder portion
    for _ in range(num_transformer_blocks):
        x = TransformerEncoder(n_heads, head_size, ff_dim, dropout)(x)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    # MLP portion for classification
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_outputs, activation='softmax')(x)
    return keras.Model(inputs, outputs)
    
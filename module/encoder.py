import tensorflow as tf
from module.utils import positional_encoding
from module.encoderLayer import EncoderLayer

class Encoder(tf.keras.layers.Layer):
        def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, encode_type="time_step"):
            super(Encoder, self).__init__()

            self.d_model = d_model
            self.num_layers = num_layers
            self.encode_type = encode_type

            self.embedding = tf.keras.layers.Dense(d_model, activation="tanh")

            self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                            for _ in range(num_layers)]

            self.dropout = tf.keras.layers.Dropout(rate)

        def call(self, x, training, mask):


            # adding embedding and position encoding.
            x = self.embedding(x)  # (batch_size, input_seq_len, d_model) or (batch_size, channel_size, d_model)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

            if self.encode_type == "time_step":
                seq_len = tf.shape(x)[1]
                self.pos_encoding = positional_encoding(seq_len,
                                                    self.d_model)
                x += self.pos_encoding[:, :seq_len, :]

            x = self.dropout(x, training=training)

            for i in range(self.num_layers):
                x = self.enc_layers[i](x, training, mask)

            return x  # (batch_size, input_seq_len, d_model)

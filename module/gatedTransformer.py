import tensorflow as tf
from module.encoder import Encoder
from module.gateLayer import GateLayer
from module.utils import create_look_ahead_mask

class GatedTransformer(tf.keras.Model):
        def __init__(self, num_layers, d_model, num_heads, dff,
                target_size, dropout=0.2):
            super().__init__()

            self.encoder_channel = Encoder(num_layers, d_model, num_heads, dff, dropout, encode_type="time_channel")

            self.encoder_step = Encoder(num_layers, d_model, num_heads, dff, dropout, encode_type="time_step")

            self.gate = GateLayer()

            self.linear = tf.keras.layers.Dense(target_size, activation="softmax")


        def call(self, inputs, training):
            # Keras models prefer if you pass all your inputs in the first argument
            time_input = inputs
            channel_input = tf.transpose(inputs, perm=[0, 2, 1])
            enc_padding_mask = None
            look_ahead_mask = create_look_ahead_mask(tf.shape(time_input)[1])


            enc_channel_output = self.encoder_channel(channel_input, training, enc_padding_mask)  # (batch_size, channel_size, d_model)
            enc_step_output = self.encoder_step(time_input, training, look_ahead_mask)  # (batch_size, inp_seq_len, d_model)

            gate = self.gate(enc_channel_output, enc_step_output)
            output = self.linear(gate)

            return output

if __name__ == "__main__":
    x = tf.random.uniform((2, 32, 4))
    print(f'input shape {x.shape}');

    num_layers = 8
    d_model = 512
    num_heads = 8
    dff = 1024

    input_size = x.shape[-1]
    channel_size = 256
    target_size = 2
    pe_input = 10000

    transformer = GatedTransformer(num_layers, d_model, num_heads, dff, input_size, channel_size,
                                    target_size, pe_input)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(32, 4)))
    model.add(transformer)
    print(model.summary())

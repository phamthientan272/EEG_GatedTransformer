import tensorflow as tf

class GateLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GateLayer, self).__init__()

        self.gate = tf.keras.layers.Dense(2)

    def call(self, x1, x2):
        x1 = tf.reshape(x1, (x1.shape[0], -1))
        x2 = tf.reshape(x2, (x2.shape[0], -1))

        x = tf.concat([x1, x2], axis=-1)
        x = self.gate(x)
        x = tf.nn.softmax(x, axis=-1)
        x = tf.concat([x1*x[..., 0:1], x2*x[..., 1:2]], axis=-1)
        return x

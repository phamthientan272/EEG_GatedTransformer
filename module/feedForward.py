import tensorflow as tf

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model: int, dff: int=512):
        super(FeedForward, self).__init__()

        self.linear1 = tf.keras.layers.Dense(dff, activation="relu")
        self.linear2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x = self.linear1(x)
        x = self.linear2(x)

        return x

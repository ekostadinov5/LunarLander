
import tensorflow as tf


class Actor(tf.keras.Model):

    def __init__(self, num_features, num_actions):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', input_shape=(1, num_features))
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_actions, activation="softmax", dtype=tf.float32)

    def call(self, x, **kwargs):
        """Forward pass"""
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x + 1e-7

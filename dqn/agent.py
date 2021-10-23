
import tensorflow as tf
import numpy as np
from dqn.model import DQN


class DQNAgent:

    def __init__(self, env):
        self.env = env

        self.num_features = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        self.main_nn = DQN(self.num_features, self.num_actions)
        self.target_nn = DQN(self.num_features, self.num_actions)

        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.mse = tf.keras.losses.MeanSquaredError()

        self.discount = 0.99

    @tf.function
    def fit(self, states, actions, rewards, next_states, dones):
        next_qs = self.target_nn(next_states)
        max_next_qs = tf.reduce_max(next_qs, axis=-1)
        targets = rewards + (1. - dones) * self.discount * max_next_qs
        with tf.GradientTape() as tape:
            qs = self.main_nn(states)
            action_masks = tf.one_hot(actions, self.num_actions)
            masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
            loss = self.mse(targets, masked_qs)
        grads = tape.gradient(loss, self.main_nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.main_nn.trainable_variables))

    def select_action(self, state):
        state_in = tf.expand_dims(state, axis=0)
        return tf.argmax(self.main_nn(state_in)[0]).numpy()

    def select_epsilon_greedy_action(self, state, epsilon):
        if tf.random.uniform((1,)) < epsilon:
            return self.env.action_space.sample()
        else:
            return self.select_action(state)

    def update_target_network(self):
        self.target_nn.set_weights(self.main_nn.get_weights())

    def save_model_weights(self, filename):
        self.main_nn.save_weights(filename)

    def load_model_weights(self, filename):
        self.main_nn(np.zeros(self.num_features).reshape((1, self.num_features)))
        self.main_nn.load_weights(filename)

        self.target_nn(np.zeros(self.num_features).reshape((1, self.num_features)))
        self.target_nn.load_weights(filename)

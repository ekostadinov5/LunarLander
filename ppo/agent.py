
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from ppo.actor import Actor
from ppo.critic import Critic


class PPOAgent:

    def __init__(self, env):
        self.env = env

        self.num_features = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        self.actor = Actor(self.num_features, self.num_actions)
        self.critic = Critic(self.num_features)
        self.actor_old = Actor(self.num_features, self.num_actions)
        self.critic_old = Critic(self.num_features)

        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        self.discount = 0.99
        self.lam = 0.95
        self.policy_kl_range = 0.0008
        self.policy_params = 20
        self.value_clip = 1.0
        self.loss_coefficient = 1.0
        self.entropy_coefficient = 0.05

    @tf.function
    def fit(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            action_probabilities, values = self.actor(states), self.critic(states)
            old_action_probabilities, old_values = self.actor_old(states), self.critic_old(states)
            next_values = self.critic(next_states)
            loss = self._get_loss(action_probabilities, values, old_action_probabilities, old_values, next_values,
                                  actions, rewards, dones)
        grads = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

    def _get_loss(self, action_probabilities, values, old_action_probabilities, old_values, next_values, actions,
                  rewards, dones):
        old_values = tf.stop_gradient(old_values)

        advantages = self._generalized_advantages_estimation(values, rewards, next_values, dones)
        returns = tf.stop_gradient(advantages + values)
        advantages = \
            tf.stop_gradient((advantages - tf.math.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-7))

        log_probabilities = self._log_probabilities(action_probabilities, actions)
        old_log_probabilities = tf.stop_gradient(self._log_probabilities(old_action_probabilities, actions))
        ratios = tf.math.exp(log_probabilities - old_log_probabilities)

        kl_divergence = self._kl_divergence(old_action_probabilities, action_probabilities)

        policy_gradient_loss = tf.where(
            tf.logical_and(kl_divergence >= self.policy_kl_range, ratios > 1),
            ratios * advantages - self.policy_params * kl_divergence,
            ratios * advantages
        )
        policy_gradient_loss = tf.math.reduce_mean(policy_gradient_loss)

        entropy = tf.math.reduce_mean(self._entropy(action_probabilities))

        clipped_values = old_values + tf.clip_by_value(values - old_values, -self.value_clip, self.value_clip)
        values_losses = tf.math.square(returns - values) * 0.5
        clipped_values_losses = tf.math.square(returns - clipped_values) * 0.5

        critic_loss = tf.math.reduce_mean(tf.math.maximum(values_losses, clipped_values_losses))
        loss = (critic_loss * self.loss_coefficient) - (entropy * self.entropy_coefficient) - policy_gradient_loss

        return loss

    def select_action(self, state, training=False):
        state_in = tf.expand_dims(state, axis=0)
        probabilities = self.actor(state_in)

        if training:
            distribution = tfp.distributions.Categorical(probs=probabilities, dtype=tf.float32)
            action = distribution.sample()
            action = int(action[0])
        else:
            action = (tf.math.argmax(probabilities, 1)[0]).numpy()

        return action

    def update_networks(self):
        self.actor_old.set_weights(self.actor.get_weights())
        self.critic_old.set_weights(self.critic.get_weights())

    def save_model_weights(self, episode):
        self.actor.save_weights("models/actor_model_" + str(episode) + ".h5")
        self.critic.save_weights("models/critic_model_" + str(episode) + ".h5")

    def load_model_weights(self, episode):
        self.actor(np.zeros(self.num_features).reshape((1, self.num_features)))
        self.actor.load_weights("models/actor_model_" + str(episode) + ".h5")
        self.actor_old(np.zeros(self.num_features).reshape((1, self.num_features)))
        self.actor_old.load_weights("models/actor_model_" + str(episode) + ".h5")

        self.critic(np.zeros(self.num_features).reshape((1, self.num_features)))
        self.critic.load_weights("models/critic_model_" + str(episode) + ".h5")
        self.critic_old(np.zeros(self.num_features).reshape((1, self.num_features)))
        self.critic_old.load_weights("models/critic_model_" + str(episode) + ".h5")

    def _generalized_advantages_estimation(self, values, rewards, next_values, dones):
        gae = 0
        advantages = []
        delta = rewards + (1.0 - dones) * self.discount * next_values - values
        for i in reversed(range(len(rewards))):
            gae = delta[i] + (1.0 - dones[i]) * self.discount * self.lam * gae
            advantages.insert(0, gae)

        return tf.stack(advantages)

    def _log_probabilities(self, action_probabilities, actions):
        distribution = tfp.distributions.Categorical(probs=action_probabilities)
        return tf.expand_dims(distribution.log_prob(actions), axis=1)

    def _kl_divergence(self, probabilities1, probabilities2):
        distribution1 = tfp.distributions.Categorical(probs=probabilities1)
        distribution2 = tfp.distributions.Categorical(probs=probabilities2)
        return tf.expand_dims(tfp.distributions.kl_divergence(distribution1, distribution2), axis=1)

    def _entropy(self, probabilities):
        distribution = tfp.distributions.Categorical(probs=probabilities)
        return distribution.entropy()

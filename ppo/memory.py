
import numpy as np


class Memory():

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def get_all_samples(self):
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int32)
        rewards = np.expand_dims(np.array(self.rewards, dtype=np.float32), axis=1)
        next_states = np.array(self.next_states, dtype=np.float32)
        dones = np.expand_dims(np.array(self.dones, dtype=np.float32), axis=1)

        return states, actions, rewards, next_states, dones

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

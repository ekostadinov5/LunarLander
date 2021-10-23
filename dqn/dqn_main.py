
import os
import numpy as np
from matplotlib import pyplot as plt
import gym
from dqn.agent import DQNAgent
from dqn.memory import ReplayBuffer
from dqn.strategy import EpsilonGreedyStrategy


# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


env = gym.make('LunarLander-v2')
agent = DQNAgent(env)


def train():
    buffer = ReplayBuffer()

    num_episodes = 50000
    epochs = 32
    batch_size = 32
    last_100_ep_reward = []
    current_frame = 0
    strategy = EpsilonGreedyStrategy(1, 0.05, 1 / num_episodes)

    training_rewards = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        epsilon = strategy.get_epsilon(episode)
        ep_reward, ep_steps, done = 0, 0, False

        while not done:
            current_frame += 1
            ep_steps += 1

            action = agent.select_epsilon_greedy_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, action, reward, next_state, done)
            ep_reward += reward

            state = next_state

            if current_frame % 2000 == 0:
                agent.update_target_network()

        if len(buffer) >= batch_size:
            for _ in range(epochs):
                states, actions, rewards, next_states, dones = buffer.get_samples(batch_size)
                agent.fit(states, actions, rewards, next_states, dones)

        print("EPISODE " + str(episode) + " - REWARD: " + str(ep_reward) + " - STEPS: " + str(ep_steps))

        if len(last_100_ep_reward) == 100:
            last_100_ep_reward = last_100_ep_reward[1:]
        last_100_ep_reward.append(ep_reward)

        if episode % 100 == 0:
            print('Episode ' + str(episode) + '/' + str(num_episodes))
            print('Epsilon: ' + str(round(epsilon, 3)))

            last_100_ep_reward_mean = np.mean(last_100_ep_reward).round(3)
            training_rewards.append(last_100_ep_reward_mean)
            print('Average reward in last 100 episodes: ' + str(last_100_ep_reward_mean))

            print()

        if episode % 10000 == 0:
            agent.save_model_weights('models/model_' + str(episode) + '.h5')

    env.close()

    plt.plot([i for i in range(100, num_episodes + 1, 100)], training_rewards)
    plt.title("Reward")
    plt.show()


def play(agent):
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
    env.close()


def evaluate():
    play(agent)

    for i in range(10000, 50000 + 1, 10000):
        agent.load_model_weights('models/model_' + str(i) + '.h5')
        print('Model trained with ' + str(i) + ' episodes')
        play(agent)

    for i in range(10):
        play(agent)


if __name__ == '__main__':
    # train()
    # evaluate()
    pass

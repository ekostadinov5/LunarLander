
import os
import numpy as np
from matplotlib import pyplot as plt
import gym
from ppo.agent import PPOAgent
from ppo.memory import Memory


# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


env = gym.make('LunarLander-v2')
agent = PPOAgent(env)


def train():
    memory = Memory()

    episode = 0
    num_episodes = 50000
    reward_threshold = 250
    threshold_reached = False
    epochs = 4
    batch_size = 32
    last_1000_ep_reward = []
    current_frame = 0

    training_rewards = []

    for episode in range(1, num_episodes + 1):

        state = env.reset()
        ep_reward, ep_steps, done = 0, 0, False

        while not done:
            current_frame += 1
            ep_steps += 1

            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            memory.add(state, action, reward, next_state, float(done))
            ep_reward += reward

            state = next_state

            if current_frame % batch_size == 0:
                for _ in range(epochs):
                    states, actions, rewards, next_states, dones = memory.get_all_samples()
                    agent.fit(states, actions, rewards, next_states, dones)
                memory.clear()
                agent.update_networks()

        print("EPISODE " + str(episode) + " - REWARD: " + str(ep_reward) + " - STEPS: " + str(ep_steps))

        if len(last_1000_ep_reward) == 1000:
            last_1000_ep_reward = last_1000_ep_reward[1:]
        last_1000_ep_reward.append(ep_reward)

        if reward_threshold:
            if len(last_1000_ep_reward) == 1000:
                if np.mean(last_1000_ep_reward) >= reward_threshold:
                    print("You solved the task after" + str(episode) + "episodes")
                    agent.save_model_weights(episode)
                    threshold_reached = True
                    break

        if episode % 1000 == 0:
            print('Episode ' + str(episode) + '/' + str(num_episodes))

            last_1000_ep_reward_mean = np.mean(last_1000_ep_reward).round(3)
            training_rewards.append(last_1000_ep_reward_mean)
            print('Average reward in last 1000 episodes: ' + str(last_1000_ep_reward_mean))

            print()

        if episode % 10000 == 0:
            agent.save_model_weights(episode)

    env.close()

    if threshold_reached:
        plt.plot([i for i in range(1000, episode, 1000)], training_rewards)
    else:
        plt.plot([i for i in range(1000, num_episodes + 1, 1000)], training_rewards)
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
        agent.load_model_weights(i)
        print('Model trained with ' + str(i) + ' episodes')
        play(agent)

    for i in range(10):
        play(agent)


if __name__ == '__main__':
    # train()
    # evaluate()
    pass

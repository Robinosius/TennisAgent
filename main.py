import csv

import gym
import numpy as np
from preprocessing import *
from agent import *

# load training data from files or create new ones

# BASIC TRAINING ALGORITHM IMPLEMENTATION
env = gym.make("Tennis-v4")
height, width, num_channels = env.observation_space.shape

height, width, num_channels = env.observation_space.shape

num_actions = 6
buffer_size = 100000
batch_size = 32
learning_rate = 1e-4
initial_epsilon = 1
final_epsilon = 0.1
epsilon_decay = (initial_epsilon / final_epsilon) / 1000000
discount_factor = 0.99

target_network_update = 10000

# center crop processor
processor = Preprocessor(84, device="cpu")

agent = Agent(84, 84, num_channels, num_actions, buffer_size, batch_size,
                    learning_rate, initial_epsilon, discount_factor, None, final_epsilon, epsilon_decay)

# Training params:
num_episodes = 100
max_steps = 1000000 # maximum number of steps per episode

episode_rewards = [] # list of rewards at end of all training episodes
episode_lengths = []

steps_total = 0

# train for number of episodes (or max number of steps, as specified)
for episode in range(num_episodes):
    # init sequence s = perception and preprocess
    obs = env.reset()

    # process image
    _obs = processor.process(obs)

    # frame stacking
    frame_stack = []
    # initialize stack with black screens
    for i in range(4):
        frame_stack.append(torch.zeros_like(_obs))

    frame_stack.pop(0)
    frame_stack.append(_obs)
    state = torch.stack(frame_stack, dim=1).squeeze(2)

    episode_reward = 0

    steps = 0
    done = False
    update = False

    while not done:
        if (steps % 1000 == 0):
            print(f"Steps in episode: {steps}")
        # select action with epsilon greedy
        action = agent.eps_greedy_selection(state)
        # execute action on environment and get next state/observation
        new_obs, reward, done, lives = env.step(action)

        episode_reward += reward

        # preprocess observation and update frame stack
        _new_obs = processor.process(obs)
        frame_stack.pop(0)
        frame_stack.append(_new_obs)
        new_state = torch.stack(frame_stack, dim=1).squeeze(2)

        # store transition (converts reward to tensor as well)
        agent.memorize(state, action, reward, new_state)

        # train agent with random minibatch
        # train agent with random minibatch
        steps += 1
        steps_total += 1

        if steps_total > training_precount:
            agent.train()

        # update target network every n steps after the end of the episode
        if steps_total%target_network_update == 0:
            agent.update_target_network()

    if update:
        agent.update_target_network()

    agent.update_eps()

    steps_total += steps
    episode_lengths.append(steps)
    episode_rewards.append(episode_reward)

    print(
        f"Episode {episode} length: {steps} mean length: {np.mean(episode_lengths)} reward: {episode_reward} mean reward: {np.mean(episode_rewards)} total step count: {steps_total}")

print("Done with training.")
model_save_name = 'tennis_agent.pt'
path = F"{model_save_name}"
agent.store_params(path)
print(f"Saved training data to {path}")

#save episode infos as csv
stats_path = 'training_statistics.csv'
with open(stats_path, 'w') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_NONE)
    wr.writerow(episode_lengths)
    wr.writerow(episode_rewards)

print(f"Saved training stats to {stats_path}")

if __name__ == "__main__":
    # prepare environment
    env = gym.make("Tennis-v4")
    height, width, channels = env.observation_space.shape
    print(height, width, channels)
    num_actions = env.action_space.n

    # commands with fire + movement make no real sense and can be discarded to reduce action space size
    print(env.unwrapped.get_action_meanings())
    height, width, channels = env.observation_space.shape  # observation space = array of pixels [r,g,b], height x width
    env.reset()
    preprocessor = Preprocessor(84)
    obs, reward, done, lives = env.step(1)
    image = preprocessor.process(obs)

    print(image)


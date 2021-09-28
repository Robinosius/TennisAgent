import gym

from preprocessing import *
from agent import *

from torch import FloatTensor

# load training data from files or create new ones

# BASIC TRAINING ALGORITHM IMPLEMENTATION
env = gym.make("Tennis-v4")
height, width, num_channels = env.observation_space.shape

num_actions = 7
buffer_size = 100000
batch_size = 32
learning_rate = 1e-4
epsilon = 1
discount_factor = 0.99

# center crop processor
processor = Preprocessor(84)

agent = TennisAgent(84, 84, num_channels, num_actions, buffer_size, batch_size,
                    learning_rate, epsilon, discount_factor)

# Training params:
num_episodes = 1

episode_count = 0
max_steps = 100000 # maximum number of steps per episode

episode_rewards = [] # list of rewards at end of all training episodes
episode_lengths = []

steps_total = 0

for episode in range(num_episodes):
    # init sequence s = perception and preprocess
    obs = env.reset()
    _obs = processor.process(obs)

    episode_reward = 0

    steps = 0
    done = False
    while not done and steps < max_steps:
        if (steps % 1000 == 0):
            print(f"Steps in episode: {steps}")
        # select action with epsilon greedy
        action = agent.eps_greedy_selection(_obs)
        # execute action on environment and get next state/observation
        new_obs, reward, done, lives = env.step(action)

        episode_reward += reward

        # print(f"Chose action: {action}")

        # preprocess observation
        _new_obs = processor.process(obs)

        # store transition (converts reward to tensor as well)
        agent.memorize(_obs, action, reward, _new_obs)

        # train agent with random minibatch
        agent.train()
        steps += 1

    steps_total += steps
    episode_lengths.append(steps)
    episode_rewards.append(episode_reward)

    print(f"Episode {episode} length: {steps} reward: {episode_reward} step count: {steps_total}")

print("Done with training.")

model_save_name = 'tennis_agent.pt'
path = F"{model_save_name}"
agent.store_params(path)
print(f"Saved training data to {path}")

#save episode infos as csv
stats_path = 'training_statistics.csv'
with open(stats_path, 'w') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
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


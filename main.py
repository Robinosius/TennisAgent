import gym

from preprocessing import *
from agent import *

from torch import FloatTensor

# BASIC TRAINING ALGORITHM IMPLEMENTATION

env = gym.make("Tennis-v4")

height, width, num_channels = env.observation_space.shape
num_actions = 1
buffer_size = 100000
batch_size = 4
learning_rate = 1
epsilon = 1
discount_factor = 1

agent = TennisAgent(height, width, num_channels, num_actions, buffer_size, batch_size,
                    learning_rate, epsilon, discount_factor)

processor = Preprocessor(84)

# Training params:
num_episodes = 9999
batch_size = 32

for episode in range(num_episodes):
    # init sequence s = perception and preprocess
    obs = env.reset()
    _obs = processor.process(obs)

    steps = 0
    done = False
    while not done:
        # select action with epsilon greedy
        action = agent.eps_greedy_selection(_obs)
        # execute action on environment and get next state/observation
        new_obs, reward, done, lives = env.step(action)
        # preprocess observation
        _new_obs = processor.process(obs)

        # store transition (converts reward to tensor as well)
        agent.memorize(_obs, action, reward, _new_obs)

        # train agent with random minibatch
        steps += 1
        agent.train()

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


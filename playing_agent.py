# agent learning to play the Atari 2600 tennis game through drl
from model import *
from replay_buffer import *
import random
import torch
from torch import optim, FloatTensor, LongTensor

# a lightweight agent that supports only playing by policy


class PlayingAgent:

    def __init__(self, height, width, num_channels, num_actions, filepath):
        # dimensions of observation space
        self.height = height
        self.width = width

        # size of action space
        self.num_actions = num_actions
        self.num_channels = num_channels

        # neural network to utilize
        self.policy_network = DQN(height, width, num_channels, num_actions)

        # load pretrained data if available
        pretrained_data = torch.load(filepath)
        self.policy_network.load_state_dict(pretrained_data)

    def select_action_policy(self, observation):
        # select an action according to learned policy
        return self.policy_network(observation.to(self.device)).max(1)[1].view(1, 1)

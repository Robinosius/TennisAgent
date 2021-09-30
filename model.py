# pytorch model for the atari dqn
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    # stack_size = number of channels for input layer
    # num_options = number of available actions that make sense for a game
    def __init__(self, num_options):
        super(DQN, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_options)

    def forward(self, x):
        # perform forward pass of the network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


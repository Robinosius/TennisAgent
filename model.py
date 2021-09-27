# pytorch model for the atari dqn
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    # stack_size = number of channels for input layer
    # num_options = number of available actions that make sense for a game
    def __init__(self, height, width, num_channels, num_options):
        super(DQN, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=8, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # calculate number of connections needed for fully connected layers
        # enables variable input image sizes
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8), 4), 3, 1)
        print(convw)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8), 4), 3, 1)
        print(convw)

        # fully connected layers
        self.fc1 = nn.Linear(convw * convh * 64, 512)
        self.fc2 = nn.Linear(512, num_options)

    def forward(self, x):
        # perform forward pass of the network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        # view =
        return self.fc2(x)


# agent learning to play the Atari 2600 tennis game through drl
from model import *
from replay_buffer import *
import random
import torch
from torch import optim, FloatTensor, LongTensor
import csv


class Agent:

    def __init__(self, height, width, num_channels, num_actions, buffer_size, batch_size,
                 learning_rate, initial_epsilon, discount_factor, filepath=None, final_epsilon=None,
                 epsilon_decay=None):
        super().__init__()
        self.random = random
        random.seed()

        # dimensions of observation space
        self.height = height
        self.width = width

        # size of action space
        self.num_actions = num_actions
        self.num_channels = num_channels

        # experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # params for learning and eps-greedy selection
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor

        # neural networks to utilize
        self.policy_network = DQN(num_actions)
        self.target_network = DQN(num_actions)

        # load pretrained data if available
        if filepath:
            pretrained_data = torch.load(filepath)
            self.policy_network.load_state_dict(pretrained_data)

        # initialize target network with values of policy network
        self.target_network.load_state_dict(self.policy_network.state_dict())

        # optimizing
        self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def eps_greedy_selection(self, observation):
        # get random number btw. 0 and 1
        choice = self.random.uniform(0, 1)
        if choice < 1 - self.epsilon:
            return self.select_action_policy(torch.tensor(observation))
        else:
            return self.select_action_random()

    def select_action_random(self):
        # randint includes both borders, num_actions should not be used as action space index
        return torch.tensor([random.randrange(self.num_actions)], device=self.device, dtype=torch.long)

    def select_action_policy(self, observation):
        # select an action according to learned policy
        return self.policy_network(observation).argmax(dim=1).to(self.device)

    def memorize(self, state, action, reward, next_state):
        # stores the memory tuple
        reward = torch.tensor([reward], device=self.device)
        self.replay_buffer.push(state, action, reward, next_state)

    def sample(self):
        return self.replay_buffer.sample(self.batch_size)

    def train(self):
        # optimize model
        # get random minibatch
        # check minibatch if there are enough elements
        if len(self.replay_buffer) < self.batch_size:
            return

        _batch = self.sample()

        batch = Transition(*zip(*_batch))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to(self.device)

        q_values = self.policy_network(states).gather(dim=1, index=actions.unsqueeze(-1))

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        expected_q_values = (next_state_values * self.discount_factor) + rewards

        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # decrease epsilon
        if self.final_epsilon and self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay

        return

    def update_target_network(self):
        print("Updating target network.")
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def store_params(self, filepath):
        torch.save(self.policy_network.state_dict(), filepath)
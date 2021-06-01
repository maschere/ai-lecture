import math, random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, batch_size):
        super(DQN, self).__init__()

        self.num_actions = num_actions
        self.memory = ReplayBuffer()
        self.batch_size = batch_size
        
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        self.optim = optim.Adam(self.parameters(), lr=self.hyperparameters["learning_rate"], eps=1e-4)
        
    def forward(self, x):
        return self.layers(x)
    
    def work(self, agentInput, epsilon):
        if random.random() > epsilon:
            agentInput = torch.from_numpy(np.array(agentInput)).float().unsqueeze(0)  # Add batch dimension with unsqueeze
            with torch.no_grad():
                q_value = self.forward(agentInput)
                action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.num_actions)
        return action


    def trainStep(self, batchSize=None):
        """
        Performs a training step for the actor and critic models, based on transitions gathered in the
        buffer. It then resets the buffer.

        :param batchSize: Overrides agent set batch size, defaults to None
        :type batchSize: int, optional
        """
        # Default behaviour waits for buffer to collect at least one batch_size of transitions
        if batchSize is None:
            if len(self.memory) < self.batch_size:
                return
            batchSize = self.batch_size
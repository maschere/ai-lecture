# %% minimal dqn example

from numpy.random import get_state
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random


#s = 125 #50
s = 110
#s = 117


random.seed(s)
np.random.seed(s)
torch.random.manual_seed(s)

# function to minimize
def f(x, y):
    if x ** 2 + y ** 2 > 2:
        return 1e3
    else:
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

# "environment"
class function_env:
    def __init__(self):
        self.x = 0.9 + random.uniform(-0.05,0.05)
        self.y = 0.9 + random.uniform(-0.05,0.05)
    def get_state(self):
        return np.array([self.x,self.y,f(self.x,self.y)])
    def step(self,selectedAction:int): #return nextSate, reward, done
        """simulate the environment for 1 "step" and return nextSate, reward"""
        #simulate our function environment...
        old_val = f(self.x,self.y)
        if selectedAction == 0:
            self.x += 0.001
        if selectedAction == 1:
            self.x -= 0.001
        if selectedAction == 2:
            self.y += 0.001
        if selectedAction == 3:
            self.y -= 0.001
        val = f(self.x,self.y)
        #calculate reward and done based on function val
        reward = -0.1
        done = 0.0
        if val < 0.00591:
            reward = 5.0
            done = 1.0
        elif val >= 100:
            reward = -5.0
            done = 1.0
        else:
            reward += (val - old_val)/100
        return (self.get_state(),reward,done)
    def reset(self):
        self.x = 0.9 + random.uniform(-0.05,0.05)
        self.y = 0.9 + random.uniform(-0.05,0.05)
        return self.get_state()

#network for the agent
def build_mlp():
    return nn.Sequential(
                nn.Linear(3, 8),
                nn.ReLU(),
                nn.Linear(8,8),
                nn.ReLU(),
                nn.Linear(8,4)
            )

#replay memory / buffer for the agent
class ReplayBuffer(deque):
    def __init__(self, capacity):
        super(ReplayBuffer,self).__init__(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self, batch_size)
        )
        return np.concatenate(state), action, reward, np.concatenate(next_state), done


class DQN_Agent:
    def __init__(self):
        self.q_net = build_mlp()
        self.target_q_net = build_mlp()
    def act(self, state, eps):
        if random.random() > eps:
            with torch.no_grad():
                state = torch.tensor(np.float32(state)).unsqueeze(0)
                q_value = self.q_net.forward(state)
                action = int(q_value.argmax(1)[0])
            return action
        else:
            return random.randrange(4)

# %% params
num_episodes = 5000
agent = DQN_Agent()
env = function_env()

buffer = ReplayBuffer(1000)
gamma = 0.99
batch_size = 32
max_steps_episode = 400
optimizer = torch.optim.Adam(agent.q_net.parameters(), lr=1e-2)

def hard_update(q_network, target_q_network):
    for t_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        if t_param is param:
            continue
        new_param = param.data
        t_param.data.copy_(new_param)
# %%
found = deque(maxlen=20)
for episode in range(num_episodes):
    epsilon =0.01 + 0.07*(1-min(episode/200,1)) #Linear annealing from 8% to 1%
    state = env.reset() ##init state
    

    for ts in range(max_steps_episode):
        action = agent.act(state, epsilon)
        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state,done)
        state = next_state
        if done > 0:
            break
    if (reward>=5):
        found.append(1.0)
    else:
        found.append(0.0)

    if len(buffer) > batch_size:
        #compute loss and update q network
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        states = torch.tensor(np.float32(states))
        next_states = torch.tensor(np.float32(next_states))
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)

        q_values = agent.q_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = agent.q_net(next_states)
        _, max_indicies = torch.max(next_q_values, dim=1)
        target_q_values = agent.target_q_net(next_states)
        next_q_value = torch.gather(target_q_values, 1, max_indicies.unsqueeze(1))
        expected_q_value = rewards + gamma * next_q_value.squeeze() * (1-done)
        loss = (q_value - expected_q_value).pow(2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"ep{episode} found: {np.mean(found)} loss {loss}")
    if episode % 2 == 0:
        hard_update(agent.q_net, agent.target_q_net)
    if np.sum(found)==found.maxlen:
        break
# %% infer loop
state = env.reset()
for ts in range(400):
    action = agent.act(state, 0)

    state, reward, done = env.step(action)
    print(f(env.x,env.y))
    if done > 0:
        print(reward)
        break

    

    
# %%

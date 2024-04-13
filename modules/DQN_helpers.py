import random
from collections import namedtuple, deque
import torch
from torch import nn, optim, tensor

# Device setup
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# device = torch.device("cpu")

# print(f"Device: {device}")
# Experience replay buffer
SARST = namedtuple('SARST', ['S', 'A', 'R', 'S_prime', 'T'])
class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer = deque([], buffer_size)
    
    def push(self, *args):
        self.buffer.append(SARST(*args))
    
    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)
    
    def __len__(self):
        return len(self.buffer)
    
# Deep Q-Network
class DQN(nn.Module):
    def __init__(self, INPUT_LAYER, LAYER1_SIZE, LAYER2_SIZE, OUTPUT_LAYER):
        super(DQN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(INPUT_LAYER, LAYER1_SIZE),
            nn.ReLU(),
            nn.Linear(LAYER1_SIZE, LAYER2_SIZE),
            nn.ReLU(),
            nn.Linear(LAYER2_SIZE, OUTPUT_LAYER)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
    
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from Runner2048 import Game

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc4b = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, output_size)
        self.leaky_relu = nn.LeakyReLU(0.01)
    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        x = torch.relu(self.fc4b(x))
        x = self.fc5(x)
        return x
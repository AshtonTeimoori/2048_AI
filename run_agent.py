import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from src.Runner2048 import Game
from model_class import DQN

# This file will be used to run the agent on the 2048 game
# The agent will be using the trained model to play the game

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 16  # Assuming the input size is 16 for the 4x4 grid of the game
output_size = 4  # Assuming there are 4 possible actions (up, down, left, right)
model = DQN(input_size, output_size).to(device)

model.load_state_dict(torch.load('2048_dqn.pth'))
model.eval()

# Create an instance of the Game class
game = Game()
#list of actions
action_dict = {0:'U', 1:'R', 2:'D', 3:'L'}

# custom_board = [[16, 4, 2, 8],
#                 [0, 2, 4, 0],
#                 [16, 256, 0, 0],
#                 [0, 0, 256, 0]]

# game.board = np.array(custom_board)
game.display()
done = False
break_counter = 0
while not done:
    state = game.get_flat_board()
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
    # if break_counter > 5:
    #     action = random.choice(range(output_size))
    
    reward, done, stuck = game.swipe(action_dict[action])
    if not stuck:
        break_counter += 1
    
    if break_counter > 5:
        done = True
        
    print('---------------------')
    game.display()
    print('Action: ', action_dict[action])
    print('Reward: ', reward)
    print('Stuck? ', not stuck)
    print('---------------------')
    # print(done)
    # print(action_dict[action])
    # print('---------------------')
print(game.get_score())
print(game.game_duration)
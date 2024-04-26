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



class ConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_vert = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(1,2), stride=1),
            nn.ReLU()
            )
        self.conv_horz = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(2,1), stride=1),
            nn.ReLU()
            )
        self.conv_vert2 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=(1,2), stride=1),
            nn.Flatten(),
            nn.ReLU()
            )
        self.conv_horz2 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=(2,1), stride=1),
            nn.Flatten(),
            nn.ReLU()
            )
    
    def forward(self, x):
        x_vert = self.conv_vert(x)
        x_horz = self.conv_horz(x)
        x_vert2vert = self.conv_vert2(x_vert)
        x_vert2horz = self.conv_horz2(x_vert)
        x_horz2vert = self.conv_vert2(x_horz)
        x_horz2horz = self.conv_horz2(x_horz)
        return torch.cat([x_vert2vert, x_vert2horz, x_horz2vert, x_horz2horz], dim=1)

class NONSQUARE(nn.Module): # 
    def __init__(self, HIDDEN_LAYER_1, HIDDEN_LAYER_2, OUTPUT_LAYER):
        super(NONSQUARE, self).__init__()
        self.network = nn.Sequential(
            ConvBlock(16, 256, 512),
            nn.ReLU(),
            nn.Flatten(),   # Unnecessary?
            nn.Linear(17408, HIDDEN_LAYER_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_1, HIDDEN_LAYER_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_2, OUTPUT_LAYER)
            )
        
    def forward(self, x):
        return self.network(x)

def encode_state(board):
  board_flat = torch.LongTensor(board)
  board_flat = nn.functional.one_hot(board_flat, num_classes=16).float().flatten()
  board_flat = board_flat.reshape(1, 4, 4, 16).permute(0, 3, 1, 2)
  return board_flat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 16  # Assuming the input size is 16 for the 4x4 grid of the game
output_size = 4  # Assuming there are 4 possible actions (up, down, left, right)

<<<<<<< HEAD
model.load_state_dict(torch.load('trained_models\duration_and_whitespace_policy_policy_weights_episode_0800.pth'))
model.eval()

# Create an instance of the Game class
game = Game(reward_type='new_values')
=======
model = NONSQUARE(2048, 256, 4).to(device)

model.load_state_dict(torch.load('C:\\Users\\Ash\\OneDrive\\Documents\\School\\GeorgiaTech\\CS7643_DeepLearning\\Project\\2048_AI\\trained_models\\hs_nonsqure_kernel_policy_policy_weights_episode_1900.pth'))
model.eval()

# Create an instance of the Game class
>>>>>>> ash
#list of actions
action_dict = {0:'U', 1:'R', 2:'D', 3:'L'}

# game.board = np.array(custom_board)
<<<<<<< HEAD
game.display()
done = False
break_counter = 0
illegalActions = []
action_counter = {'U':0, 'R':0, 'D':0, 'L':0}
while not done:
    state = game.get_flat_board()
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    with torch.no_grad():
            q_values = model(state_tensor)
            q_values[illegalActions] = torch.tensor(float('-inf')).to(device)
            best_legal_action = q_values.argmax().item()
            action = best_legal_action
    # if break_counter > 2:
    #     action = random.choice(range(output_size))
    reward, done, stuck, A, _, illegalActions = game.swipe(action_dict[action])
    if not stuck:
        break_counter += 1
    
    if break_counter > 2:
        done = True
    action_counter[action_dict[action]] += 1
    print('---------------------')
    print('Action: ', action_dict[action])
    print('Reward: ', reward)
    print('Stuck? ', not stuck)
    print('Q-values: ', q_values)
    game.display()
    print('---------------------')
    # print(done)
    # print(action_dict[action])
    # print('---------------------')
print('Final score: ', game.get_score())
print('Moves: ', game.game_duration)
print('Action counter: ', action_counter)
=======
iterations = 100
max_tiles = np.zeros([iterations])

game = Game(np.random.randint(1, 1000), 4, 'hs')
for epoch in range(iterations):
    game.reset()
    # game.display()
    done = False
    invalid_actions = []

    while not done:
        # Get the state from the model
        state = encode_state(game.get_flat_board()).flatten()
        state = torch.tensor(state, dtype=torch.float32, device=device)

        # Get the action from the model
        with torch.no_grad():
                actions = model(state.view(1,16,4,4))
                for invalid in invalid_actions:
                    actions[0, invalid] = -torch.inf
                action = actions.argmax().item()

        (reward, done, updated, invalid_actions, invalid_moves_made) = game.swipe(action_dict[action])
        
        # print(reward)
        
    game.display()
    
    max_tiles[epoch] = np.max(game.board)
    # print(done)
    # print(action_dict[action])
    # print('---------------------')
# print(game.get_score())
# print(game.game_duration)
plt.hist(max_tiles, bins=range(int(np.max(max_tiles))+2))
plt.xlabel('Max Tiles')
plt.ylabel('Frequency')
plt.title('Distribution of Max Tiles')
plt.show()
>>>>>>> ash

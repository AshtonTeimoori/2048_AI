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
from generate_board import generate_2048_games

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Create an instance of the Game class
# game = Game(reward_type='no_shaping')
game = Game(reward_type='duration_and_whitespace')
#list of actions
action_dict = {0:'U', 1:'R', 2:'D', 3:'L'}
# state = game.reset()

# Define hyperparameters
input_size = 16  # Assuming the input size is 16 for the 4x4 grid of the game
output_size = 4  # Assuming there are 4 possible actions (up, down, left, right)
LR = 0.01
matches = 600
GAMMA = 0.8 # Discount factor
TAU = 0.1 # Soft update parameter
EPS = 0.9 # Epsilon greedy parameter
EPS_DECAY = 20000
EPS_MIN = 0.01
BATCH_SIZE = 1024

memory = ReplayMemory(10000)

# Create an instance of the DQN model
policy_net = DQN(input_size, output_size).to(device)
target_net = DQN(input_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())

# Define the loss function and optimizer

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=False)

steps = 0
eps = []
def select_action(state):
    sample = random.random()
    global steps
    global eps 
    eps.append(EPS_MIN + (EPS - EPS_MIN) * math.exp(-1 * steps / EPS_DECAY))
    eps_thresh = eps[-1]
    steps += 1
    if sample > eps_thresh:
        with torch.no_grad():
            q_values = policy_net(state)
            action = torch.argmax(q_values).item()
    else:
        action = random.choice(range(output_size))
    return action

def optimize_model(state, reward):
    
    if len(memory) < BATCH_SIZE:
        return
    
    # batch the memory
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(device)
    
    state_batch = torch.cat(batch.state).reshape(BATCH_SIZE, -1)
    action_batch = torch.cat(batch.action).reshape(BATCH_SIZE, -1)
    reward_batch = torch.cat(batch.reward).reshape(BATCH_SIZE, 1)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
    next_state_values = next_state_values.unsqueeze(1)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    
    loss = criterion(state_action_values, expected_state_action_values)
    
    optimizer.zero_grad()
    loss.backward()
    
    # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)
    optimizer.step()
    
    return loss.item()

game_history_vect = []
loss_vect = [0]
duration_vect = []
score_vect = []

q_value_vect = []

generate_maps = False

# Training loop
if generate_maps == True:
    # this will randomly generate a list of games to train on up to tiles of 10**10
    # this will train on games for only the first action taken
    
    for episode in range(matches):
        # state = game.reset()  # Reset the game and get the initial state
        games = generate_2048_games(4, 1000)
        for game in games:
            # Convert the state to a tensor 
            state = game.get_flat_board()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            
            # Choose an action based on the Q-values (e.g., using epsilon-greedy policy)
            action = select_action(state_tensor)

            # Take the chosen action and get the next state, reward, and done flag
            reward, done, _ = game.swipe(action_dict[action])
                
            reward = torch.tensor([reward], device=device)
            
            if done:
                next_state = None
                break
            else:
                next_state = game.get_flat_board()
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            
            action = torch.tensor([action], device=device, dtype=torch.long)
            # Store the transition in the replay memory
            memory.push(state_tensor, action, next_state, reward)
            
            # Update the current state
            state_tensor = next_state
            
            loss = optimize_model(state_tensor, reward)
            
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            
            # game.display()
            
            
        duration_vect.append(game.game_duration)
        game_history_vect.append(np.sum(game.get_flat_board()))
        if loss is not None:
            loss_vect.append(loss)
        # print(duration_vect[-1])
        print(game_history_vect[-1])
        game.display()
        if episode % 100 == 0:
            print('---------------------')
            print('Episode:', episode)
            print('---------------------')
            
else:
    # this will train on a single game from a starting board up to game end
    # there's also a break counter to stop the game early if desired (to train on only the early game for example)
    for episode in range(matches):
        state = game.reset()  # Reset the game and get the initial state
        done = False  # Flag to indicate if the game is over
        # Convert the state to a tensor 
        # state = game.get_flat_board()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        
        break_counter = 0
        while not done:
            # Get the Q-values from the model
            # q_values = model(state_tensor)
            
            # Choose an action based on the Q-values (e.g., using epsilon-greedy policy)
            action = select_action(state_tensor)

            # Take the chosen action and get the next state, reward, and done flag
            reward, done, stuck = game.swipe(action_dict[action])
            
            if stuck:
                break_counter += 1
                
            # if break_counter > 100:
            #     done = True
                
            reward = torch.tensor([reward], device=device)
            
            if done:
                next_state = None
                state = game.get_flat_board()
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    q_value_vect.append(q_values)
                break
            else:
                next_state = game.get_flat_board()
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            
            action = torch.tensor([action], device=device, dtype=torch.long)
            # Store the transition in the replay memory
            memory.push(state_tensor, action, next_state, reward)
            
            
            # Update the current state
            state_tensor = next_state
            
            
                
            # if episode % 10 == 0:
            loss = optimize_model(state_tensor, reward)
            
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            
            # game.display()
            
            
        
        duration_vect.append(game.game_duration)
        game_history_vect.append(np.sum(game.get_flat_board()))
        if loss is not None:
            loss_vect.append(loss)
            
        print('---------------------')
        print(duration_vect[-1])
        print('Q-values:', q_value_vect[-1])
        print('Loss:', loss_vect[-1])
        score_vect.append(game.get_score())
        # print(game_history_vect[-1])
        game.display()
        print('---------------------')
        if episode % 100 == 0:
            print('---------------------')
            print('Episode:', episode)
            print('---------------------')


torch.save(policy_net.state_dict(), '2048_dqn.pth')

smooth = 100
fig, axs = plt.subplots(2, 2)

smoothed_duration = np.convolve(duration_vect, np.ones(smooth)/smooth, mode='valid')
axs[0, 0].plot(smoothed_duration)
axs[0, 0].set_title('Game Duration')

smoothed_loss = np.convolve(loss_vect, np.ones(smooth)/smooth, mode='valid')
axs[0, 1].plot(smoothed_loss)
axs[0, 1].set_title('Loss')

smoothed_score = np.convolve(score_vect, np.ones(smooth)/smooth, mode='valid')
axs[1, 0].plot(smoothed_score)
axs[1, 0].set_title('Score')

smoothed_eps = np.convolve(eps, np.ones(smooth)/smooth, mode='valid')
axs[1, 1].plot(smoothed_eps)
axs[1, 1].set_title('Epsilon')

plt.tight_layout()
plt.savefig('figures/subplots.png')
plt.show()

game.reset()
# game.display()
# done = False
# break_counter = 0
# while not done:
#     state = game.get_flat_board()
#     state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
#     with torch.no_grad():
#             q_values = policy_net(state_tensor)
#             action = torch.argmax(q_values).item()
#     reward, done, stuck = game.swipe(action_dict[action])
#     if not stuck:
#         break_counter += 1
    
#     if break_counter > 5:
#         done = True
#     print(reward)
#     game.display()
#     # print(done)
#     # print(action_dict[action])
#     # print('---------------------')
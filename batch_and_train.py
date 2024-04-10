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
import multiprocessing
from multiprocessing.managers import BaseManager

# ignore divide by zero warning
# the log2 calc on the first step of the training will result in a divide by zero warning
np.seterr(divide='ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemoryPar(object):

    def __init__(self, shared_list):
        self.memory = shared_list

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class MyManager(BaseManager):
    pass


MyManager.register('ReplayMemoryPar', ReplayMemoryPar)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, stack):
        """Save a transition"""
        self.memory.extend(stack)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

action_dict = {0:'U', 1:'R', 2:'D', 3:'L'}
# state = game.reset()

# Define hyperparameters
input_size = 16  # Assuming the input size is 16 for the 4x4 grid of the game
output_size = 4  # Assuming there are 4 possible actions (up, down, left, right)
LR = 0.01
LR_DECAY = 0.01
matches = 25
GAMMA = 0.99 # Discount factor
TAU = 0.1 # Soft update parameter
EPS = 0.99 # Epsilon greedy parameter
EPS_DECAY = 100
EPS_MIN = 0.1
BATCH_SIZE = 1024
games_to_gen = 10

# Create an instance of the DQN model
policy_net = DQN(input_size, output_size).to(device)
target_net = DQN(input_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())

# for taking actions on the parallel cpus
cpu_policy_net = DQN(input_size, output_size)
cpu_policy_net.load_state_dict(policy_net.state_dict())

# Define the loss function and optimizer
criterion = nn.SmoothL1Loss().to(device)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

steps = 0
def select_action(state, episode):
    sample = random.random()
    global steps
    eps_thresh = EPS_MIN + (EPS - EPS_MIN) * math.exp(-1 * episode / EPS_DECAY)
    steps += 1
    if sample > eps_thresh:
        with torch.no_grad():
            q_values = cpu_policy_net(state)
            action = torch.argmax(q_values).item()
    else:
        action = random.choice(range(output_size))
    return action

def run_game(rew_struct, shared_list, episode):
    memory = ReplayMemoryPar(shared_list)
    game = Game(reward_type=rew_struct)
    state = game.get_flat_board()
    done = False
    state_tensor = torch.tensor(state, dtype=torch.float32)
    state_vect = []
    action_vect = []
    next_state_vect = []
    reward_vect = []
    
    while not done:
        # Perform a random action
        action = select_action(state_tensor, episode)
        reward, done, stuck = game.swipe(action_dict[action])
        
        reward = torch.tensor([reward])
        
        if done:
            next_state = None
            
            for i in range(len(state_vect)):
                memory.push(state_vect[i], action_vect[i], next_state_vect[i], reward_vect[-1])
            break
        else:
            next_state = game.get_flat_board()
            next_state = torch.tensor(next_state, dtype=torch.float32)
        
        action = torch.tensor([action], dtype=torch.long)
        # Store the transition in the replay memory
        state_vect.append(state_tensor)
        action_vect.append(action)
        next_state_vect.append(next_state) 
        reward_vect.append(reward)
        # memory.push(state_tensor, action, next_state, reward)
        
        state_tensor = next_state
            
            
def optimize_model(memory):
    
    if len(memory) < BATCH_SIZE:
        return
    
    # batch the memory
    # transitions = random.sample(memory, BATCH_SIZE)
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(device)
    
    state_batch = torch.cat(batch.state).reshape(BATCH_SIZE, -1).to(device)
    action_batch = torch.cat(batch.action).reshape(BATCH_SIZE, -1).to(device)
    reward_batch = torch.cat(batch.reward).reshape(BATCH_SIZE, -1).to(device)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = criterion(state_action_values, expected_state_action_values)
    
    optimizer.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)
    optimizer.step()
    
    return loss.item()


game_history_vect = []
loss_vect = []
duration_vect = []
score_vect = []

if __name__ == '__main__':
    memory = ReplayMemory(10000)
    for episode in range(matches):
        
        # run games in parallel
        with multiprocessing.Manager() as manager:
            shared_list = manager.list()
            # Define a function to run a game and store the resulting states in memory
            

            # Create a list to hold the game processes
            processes = []

            # Run 20 games from Game in parallel
            for _ in range(games_to_gen):
                # game ='only_duration'
                # game = 'no_shaping'
                game = 'only_steps'
                
                # Create a process for each game and start it
                process = multiprocessing.Process(target=run_game, args=(game, shared_list, episode))
                process.start()
                
                # Add the process to the list
                processes.append(process)

            # Wait for all processes to finish
            for process in processes:
                process.join()
            
            new_runs = list(shared_list)
            memory.push(new_runs)
            # memory = list(shared_list)
            print(len(memory))
            
        loss = optimize_model(memory)
        
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
        # copy current policy net to cpu_policy_net for choosing actions
        cpu_policy_net.load_state_dict(policy_net.state_dict())
        
        # game = Game()
        if loss is not None:
            loss_vect.append(loss)
        if episode % 1 == 0:
            print('---------------------')
            print('Episode:', episode)
            print('Loss:', loss)
            print('---------------------')
    torch.save(policy_net.state_dict(), '2048_dqn.pth')
    

# smoothed_loss = np.convolve(loss_vect, np.ones(smooth)/smooth, mode='valid')

    plt.plot(loss_vect)
    # plt.show()
    plt.savefig('figures/loss.png')
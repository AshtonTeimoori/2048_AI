# Description: This file contains the implementation of the DQN algorithm for the 2048 game.
#

# Imports
import matplotlib
import matplotlib.pyplot as plt
from math import exp
import numpy as np
from itertools import compress
import time 
import json
from src.Runner2048 import Game
from modules.DQN_helpers import DQN, ReplayBuffer, SARST
from torch import nn, optim, tensor
import torch
import random
# from IPython import display

# Device setup
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# # device = torch.device("cpu")

# print(f"Device: {device}")

# Epsilon greedy policy
def epsilonGreedy(state, network, nA, epsilon):
    # Decide whether to explore or exploit
    greedy = (random.random() > epsilon)
    
    if random.random() > epsilon:
        # Pick best action, if tie, use lowest index
        with torch.no_grad():   # Speeds up computation
            return network(state).argmax().item()
    else:
        return torch.tensor([[random.randrange(nA)]], dtype=torch.long).item()
    
# step = 0
def getEpsilon(step):
    # global step
    epsilon = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * exp(-1. * step / EPSILON_DECAY)
    step += 1
    return step, epsilon

# Environment setup
env = Game(seed=1, board_size=4, reward_type='duration_and_largest')
action_dict = {0:'U', 1:'R', 2:'D', 3:'L'}

# Parameters
nS = 16
nA = 4

# Hyperparameters
BATCH_SIZE = 2**8
LAYER1_SIZE = 2**8
LAYER2_SIZE = 2**8
EPISODES_TRAINING = 20000
ALPHA = 1e-1
GAMMA = 0.99
TAU = 1e-2
EPSILON_MAX = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 50
BUFFER_SIZE = 10000

# Setup
policy_net = DQN(nS, LAYER1_SIZE, LAYER2_SIZE, nA)
target_net = DQN(nS, LAYER1_SIZE, LAYER2_SIZE, nA)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA, amsgrad=True)
RB = ReplayBuffer(BUFFER_SIZE)

def train():
    minibatch_awk = RB.sample(BATCH_SIZE)
    minibatch = SARST(*zip(*minibatch_awk))

    N = len(minibatch.S)

    S = torch.cat(minibatch.S)
    A = minibatch.A
    R = torch.cat(minibatch.R)
    maxQ = torch.zeros(N, 1)
    nonterm_mask = tensor(minibatch.T)

    Q_SA = policy_net(S).gather(1, torch.reshape(tensor(A), [N, 1]))

    with torch.no_grad():
        S_prime_masked = list(compress(minibatch.S_prime, minibatch.T))
        maxQ[nonterm_mask] = torch.reshape(target_net(torch.cat(S_prime_masked)).max(1)[0], [sum(nonterm_mask).item(), 1])

    y = (maxQ * GAMMA) + R.unsqueeze(1)

    # Compute loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(Q_SA, y)
    # criterion = nn.HuberLoss(delta=0.95)
    # loss = criterion(Q_SA, y)
    # criterion = nn.MSELoss()
    # loss = criterion(Q_SA, y)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    #In-place gradient clipping
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()

# Plotting
def plot_multi(title, axis, args, save_string=""):
    n_plots = len(args)
    plt.clf()
    fig, ax = plt.subplots(n_plots, 1, sharex=True)
    for argi, arg in enumerate(args):
        data = torch.tensor(arg, dtype=torch.float)
        ax[argi].set_title(title[argi])
        ax[argi].set_ylabel(axis[argi])
        ax[argi].plot(data)

        if len(arg) >= 100:
            means = data.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            ax[argi].plot(means.numpy())
        
    plt.xlabel("Episode")
    plt.pause(0.001)
    # display.clear_output(wait=True)   # notebook
    if save_string != "":
        plt.savefig("./figures/"+save_string+".png")

# Training
episodic_rewards = []
episodic_losses = []
episodic_epslions = []
episdoic_duration = []

def DQN_network(episodes):
    start_time = time.time()
    T = 0
    step = 0
    for epi in range(episodes):
        S = env.reset()
        S = torch.tensor([S], dtype=torch.float32)
        
        episodic_reward = 0
        episodic_mean_loss = 0
        terminated = False

        step, epsilon = getEpsilon(step)

        while not terminated:
            T += 1
            
            # Choose action
            A = epsilonGreedy(S, policy_net, nA, epsilon)
            # Take action
            reward, terminated, updated = env.swipe(action_dict[A])
            S_prime = env.get_flat_board()
            S_prime = [0] if terminated else tensor([S_prime], dtype=torch.float32)

            # Store transition in replay buffer
            RB.push(S, A, tensor([reward], dtype=torch.float32), 
                    S_prime, tensor(not terminated, dtype=torch.bool))
            
            S = S_prime

            # Update the networks
            if len(RB) > BATCH_SIZE:
                episodic_mean_loss += train()
            
            episodic_reward += reward

            if T%10 == 0:
                # Soft update of the target network's weights
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

            if epi % 10 == 0:
                env.display()
                time.sleep(0.1)

        episodic_losses.append(episodic_mean_loss/T)
        episodic_rewards.append(episodic_reward)
        episodic_epslions.append(epsilon)
        episdoic_duration.append(env.game_duration)

        # Saving the model (maybe)
        if epi % 100 == 0:
            save_string = "_policy_weights_episode_"+str(epi).zfill(4)
            torch.save(target_net.state_dict(), "./trained_models/target_net"+save_string+".pth")
            torch.save(policy_net.state_dict(), "./trained_models/policy_net"+save_string+".pth")
        if epi % 10 == 0:
            print(f"Episode: {epi}, Reward: {episodic_reward}, Duration: {env.game_duration}, Loss: {episodic_mean_loss/T}, Epsilon: {epsilon}")
            
            print()
            print()
            print()
            print()
            print()
            
            # plot_multi(["Reward History", "Loss History", "Epsilon History", "Duration History"], 
            #         ["Reward", "Loss", "Epsilon", "Duration"], 
            #         [episodic_rewards, episodic_losses, episodic_epslions, episdoic_duration], 
            #         save_string="DQN_training")
            # plt.show()
            # plot_multi(["Reward", "Loss", "Epsilon", "Duration"], ["Reward", "Loss", "Epsilon", "Duration"], [episodic_rewards, episodic_losses, episodic_epslions, episdoic_duration], "DQN_training")
        
    delta_time = time.time() - start_time
    plot_multi(["Reward History", "Loss History", "Duration History", "Epsilon History"], 
               ["Reward", "Loss", "Duration", "Epsilon"], 
               [episodic_rewards, episodic_losses, episdoic_duration, episodic_epslions], 
               save_string="DQN_training")
    # plt.ioff()
    plt.show()

DQN_network(EPISODES_TRAINING)

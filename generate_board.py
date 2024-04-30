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



def generate_2048_games(size=4, num_games=1000):
    # randomly generate 2048 game boards for training
    # size: size of the board
    # num_games: number of games to generate
    # return: list of Game objects
    # Example:
    # games = generate_2048_games(4, 1000)
    # print(games[0].board)
    # note: this was haphazardly written and may not generate very logical 2048 games
    games = []
    for i in range(num_games):
        # game = Game(board_size=size)
        # game.board = np.exp2(np.random.randint(0, 10, (size, size)))
        
        # game.board = np.where(game.board == 1, 0, game.board)
        board = np.random.choice([0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], size=(size, size), replace=True, p=[0.30, 0.2, 0.16, 0.1, 0.08, 0.06, 0.04, 0.02, 0.02, 0.01, 0.01]) # randomly generate 2048 game boards
        games.append(board)
    return games


# games = generate_2048_games(4, 1)

# print(games[0].board)
import numpy as np
import random
import math
from rewards import reward_selection
# import pandas as pd
class Game:
    def __init__(self, seed=None, board_size=4, reward_type='no_shaping'):
        self.seed = seed
        random.seed(seed)
        self.largest_value = 0

        self.board_size = board_size
        self.board = np.zeros([board_size, board_size], dtype=int)
        self.previous_board = np.zeros([board_size, board_size], dtype=int)

        self.prob_4 = 0.1   # Probability that a 4 will spawn

        self.score = 0
        self.game_over = False

        self.game_duration = 0
        
        self.empty_space_val = self.board_size**2   # Number of empty spaces on the board

        self.reward_type = reward_type

        self.created_val = 0
        self.created_val_count = 0


        self.setup()

    def reset(self):
        
        # random.seed(self.seed)
        random.seed(random.randrange(2**63-1))
        self.board = np.zeros([self.board_size, self.board_size], dtype=int)
        self.previous_board = np.zeros([self.board_size, self.board_size], dtype=int)
        self.largest_value = 0


        self.prob_4 = 0.1   # Probability that a 4 will spawn

        self.score = 0
        self.game_over = False

        self.game_duration = 0
        
        self.empty_space_val = self.board_size**2
        self.setup()
        self.created_val = 0
        self.created_val_count = 0
        
        return self.get_flat_board()

    def load_board(self, board, game_duration):
        self.board = board
        self.previous_board = np.zeros([self.board_size, self.board_size], dtype=int)

        self.prob_4 = 0.1   # Probability that a 4 will spawn

        self.score = sum(sum(self.board)).item()
        self.game_over = False

        self.game_duration = game_duration
        self.empty_space_val = len(self.get_avaliable_spaces())
        
        return self.get_flat_board()

    def save_off_board(self):
        return (self.board, self.game_duration)

    def setup(self):
        for i in range(2):
            self.add_tile()
    
    def add_tile(self):
        self.add_tile_to_board(self.get_new_tile_pos(), self.get_new_tile_val())

    def get_avaliable_spaces(self):
        avaliable = []
        counter = 0
        for board_row in self.board:
            for cell_val in board_row:
                if cell_val == 0:
                    avaliable.append(counter)
                counter = counter + 1
        return avaliable
    
    def add_tile_to_board(self, pos, val):
        # 
        self.board[pos[0], pos[1]] = val
    
    def get_new_tile_pos(self):
        # 
        pos = random.choice(self.get_avaliable_spaces())
        row = int(pos / self.board_size)
        col = pos % self.board_size
        return [row, col]
    
    def get_new_tile_val(self):
        # 
        val = 2
        if (random.random() <= self.prob_4):
            val = 4
        return val
    
    def display(self):
        print(self.board)
        # self.print_csi()

    def swipe(self, dir):
        # Note: 
        #   Axis == 0 -> all the values in their respective column
        #   Axis == 1 -> all the values in their respective row
        #   If we are swiping to the right (moving pieces to the right) 
        #       1. Start on the right side and loop through moving pieces to the right
        #       2. Check for combinations when moving the pieces to the right
        self.previous_board = np.copy(self.board)
        largest_created_val = 0
        
        updated = False
        updated_i = False

        valid_move_only = False

        valid_move_toggle = True

        action_dict = {0:'U', 1:'R', 2:'D', 3:'L'}
        action_dict_rev = {'U':0, 'R':1, 'D':2, 'L':3}

        invalid_moves_made = 0

        self.created_val = 0
        self.created_val_count = 0

        # Move the piece 
        # Check direction:

        if (dir == 'R' or dir == 'D'):
            rev = True
        else:
            rev = False

        if (dir == 'L' or dir == 'R'):
            # 'L' (Left) or R (Right) - see if there are any empty rows
            for row in range(self.board_size):
                (self.board[row, :], updated_i, largest_created_val_i) = self.updated_rowcol(self.board[row, :], rev)
                if largest_created_val_i > largest_created_val:
                    largest_created_val = largest_created_val_i
                if updated_i:
                    updated = True

        elif(dir == 'U' or dir == 'D'):
            # 'U' (Up) or 'D' (Down) - see if there are any empty columns
            for col in range(self.board_size):
                (self.board[:, col], updated_i, largest_created_val_i) = self.updated_rowcol(self.board[:, col], rev)
                if largest_created_val_i > largest_created_val:
                    largest_created_val = largest_created_val_i
                if updated_i:
                    updated = True

        else:
            print("Bad Swipe Direction")
            exit(1)

        if updated:
            self.add_tile()
            self.check_gameover()
            self.game_duration += 1
            self.largest_value = max(self.largest_value, np.max(self.board))
        else:
            invalid_moves_made += 1

        # Create an array of invalid moves
        update_L = False
        update_R = False
        update_U = False
        update_D = False

        invalid_moves = []
        for rowcol in range(self.board_size):
            # Check if we can move right
            (_, updated_i, _) = self.updated_rowcol(self.board[rowcol, :], True, True)
            update_R = update_R or updated_i
            # Check if we can move left
            (_, updated_i, _) = self.updated_rowcol(self.board[rowcol, :], False, True)
            update_L = update_L or updated_i
            # Check if we can move down
            (_, updated_i, _) = self.updated_rowcol(self.board[:, rowcol], True, True)
            update_D = update_D or updated_i
            # Check if we can move up
            (_, updated_i, _) = self.updated_rowcol(self.board[:, rowcol], False, True)
            update_U = update_U or updated_i
        
        if not update_L:
            invalid_moves.append(action_dict_rev['L'])
        if not update_R:
            invalid_moves.append(action_dict_rev['R'])
        if not update_U:
            invalid_moves.append(action_dict_rev['U'])
        if not update_D:
            invalid_moves.append(action_dict_rev['D'])

        reward = reward_selection(self, updated, largest_created_val)

        return (reward, self.game_over, updated, invalid_moves, invalid_moves_made)
    
    def updated_rowcol(self, cur_rowcol, rev, check_only=False):
        #   If we are swiping to the left (moving pieces to the left) 
        #       1. Start on the left side and loop through moving pieces to the left
        #       2. Check for combinations when moving the pieces to the left
        updated = False
        largest_created_val = 0

        new_rowcol = np.zeros(self.board_size, dtype=int)

        if rev:
            cur_rowcol = np.flip(cur_rowcol)

        last_val = 0
        new_index = 0

        for i in range(self.board_size):
            if cur_rowcol[i] == 0:
                # If its empty, just keep going forward
                continue
            else:
                # If the current value is a non-zero, move it to the front/see if we can combine
                if cur_rowcol[i] == last_val:
                    # If the last value we put down is the same as this one, combine
                    new_index -= 1
                    new_rowcol[new_index] = 2*last_val
                    last_val = 0
                    updated = True
                    self.score += new_rowcol[new_index]
                    if new_rowcol[new_index] > largest_created_val:
                        largest_created_val = new_rowcol[new_index]
                    if not check_only:
                        self.created_val += new_rowcol[new_index]
                        self.created_val_count += 1
                else:
                    # New value to put down 
                    new_rowcol[new_index] = cur_rowcol[i]
                    last_val = cur_rowcol[i]
                    if (new_index != i):
                        updated = True
                new_index += 1

        # Flip it back to normal
        if (rev):
            new_rowcol = np.flip(new_rowcol)

        return (new_rowcol, updated, largest_created_val)

    def get_score(self):
        return self.score
        # return np.sum(self.board)

    def get_flat_board(self):
        log_board = np.copy(self.board)
        log_board[self.board > 0] = np.log2(self.board[self.board > 0])
        # return log_board.flatten()/np.log2(2048) 
        # return log_board.flatten()/np.max(log_board)
        # return log_board.flatten()/16
        return log_board.flatten()
        # return self.board.flatten()#/np.max(log_board)
    
    def get_plump_board(self):
        log_board = np.copy(self.board)
        log_board[self.board > 0] = np.log2(self.board[self.board > 0])
        # return log_board.flatten()/np.log2(2048) 
        return log_board/16
    
    def check_gameover(self):
        for rowcol in range(self.board_size):
                (_, got_moves, _) = self.updated_rowcol(self.board[:, rowcol], False, True)
                if got_moves: return
                (_, got_moves, _) = self.updated_rowcol(self.board[:, rowcol], True, True)
                if got_moves: return
                (_, got_moves, _) = self.updated_rowcol(self.board[rowcol, :], False, True)
                if got_moves: return
                (_, got_moves, _) = self.updated_rowcol(self.board[rowcol, :], True, True)
                if got_moves: return
        
        self.game_over = True

    def print_csi(self):
        csi_up = f"\x1B[{5}A"
        csi_clr= "\x1B[0K"
        
        print(f'{csi_up}{csi_clr}')
        for r in range( 0, len(self.board) ): 
            print(f'{csi_clr}{self.board[r][0]}\t{self.board[r][1]}\t{self.board[r][2]}\t{self.board[r][3]}{csi_clr}')

        # csi_up = f"\x1B[{5}A"
        # csi_clr = "\x1B[2K"  # Clear entire line
            
        # print(f'{csi_up}{csi_clr}')
        # for r in range(len(self.board)):
        #     print(f'{self.board[r][0]}\t{self.board[r][1]}\t{self.board[r][2]}\t{self.board[r][3]}{csi_clr}')



# def test(seed, board_size, move_list):
#     game = Game(seed, board_size)
#     game.display()

#     for move in move_list:
#         game.swipe(move)
#         game.display()

# test(420, 4, ['R', 
#               'U',
#               'U',
#               'U',
#               'D',
#               'L',
#               'R',
#               'U',
#               'D',
#               'L',
#               'D',
#               'L',
#               'U',
#               'D',
#               'D',
#               'L',
#               'L',
#               'D',
#               'L',
#               'D',
#               'L',
#               'D',
#               'L',
#               'D'])
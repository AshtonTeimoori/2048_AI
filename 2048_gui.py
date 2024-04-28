import numpy as np
import random
import pyglet
from pyglet.window import key
from pyglet import shapes
import math
from rewards import reward_selection

class Game:
    def __init__(self, seed, board_size, reward_type):
        random.seed(seed)

        self.board_size = board_size
        self.board = np.zeros([board_size, board_size], dtype=int)

        self.prob_4 = 0.1   # Probability that a 4 will spawn

        self.score = 0
        self.game_over = False

        self.reward_type = reward_type

        self.setup()

        # self.game_window = pyglet.window.Window()

    def setup(self):
        for i in range(2):
            self.add_tile()
        
        # Starting game
        self.start_game()
    
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
    
    def start_game(self):

        self.game_window = pyglet.window.Window()

        window_size = 500
        self.game_window .set_size(window_size, window_size)

        @self.game_window.event
        def on_key_press(symbol, modifiers):
            if symbol == key.LEFT:
                # print('The left arrow key was pressed.')
                # self.label_text='The left arrow key was pressed.'
                reward, game_over_, updated_ = self.swipe('L')
                print("Reward: ", reward)
                # print("Game Over: ", game_over_)
                # print("Good Move: ", updated_)
                self.display()
            elif symbol == key.RIGHT:
                # print('The right arrow key was pressed.')
                # self.label_text='The right arrow key was pressed.'
                reward, game_over_, updated_ = self.swipe('R')
                print("Reward: ", reward)
                # print("Game Over: ", game_over_)
                # print("Good Move: ", updated_)
                self.display()
            elif symbol == key.UP:
                # print('The up arrow key was pressed.')
                # self.label_text='The up arrow key was pressed.'
                reward, game_over_, updated_ = self.swipe('U')
                print("Reward: ", reward)
                # print("Game Over: ", game_over_)
                # print("Good Move: ", updated_)
                self.display()
            elif symbol == key.DOWN:
                # print('The down arrow key was pressed.')
                # self.label_text='The down arrow key was pressed.'
                reward, game_over_, updated_ = self.swipe('D')
                print("Reward: ", reward)
                # print("Game Over: ", game_over_)
                # print("Good Move: ", updated_)
                self.display()
            elif symbol == key.ENTER:
                print('The enter key was pressed.')
                self.label_text='The enter key was pressed.'
            elif symbol == key.Q:
                print('The \'q\' key was pressed.')
                self.label_text='The \'q\' key was pressed.'


        @self.game_window.event
        def on_draw():
            self.game_window.clear()
            self.gui_structure_board()

        pyglet.app.run()

    def gui_structure_board(self):
        batch = pyglet.graphics.Batch()
        min_dim = min(self.game_window.width, self.game_window.height)
        step_size = min_dim//(self.board_size+1)

        box_board = 5

        for row in range(self.board_size):
            for col in range(self.board_size):
                label = pyglet.text.Label(str(self.board[row, col]),
                                    font_name='Times New Roman',
                                    font_size=36,
                                    x=step_size*(col+1), y=min_dim-step_size*(row+1),
                                    anchor_x='center', anchor_y='center')
                
                if (self.board[row, col] == 0):
                    color = (205, 205, 205)
                else:
                    # max_color = 2**(self.board_size**2-1)
                    max_color = 2**14
                    # mix = int(((255-125)/math.log(max_color,2))*math.log(self.board[row, col],2))
                    mix = int(((255-0)/math.log(max_color,2))*math.log(self.board[row, col],2))
                    color = (255, 255-mix, 0)
                square = shapes.Rectangle(box_board/2+step_size*(col+1)-step_size/2, 
                                          box_board/2+min_dim-step_size*(row+1)-step_size/2, 
                                          step_size-box_board/2, step_size-box_board/2, 
                                          color=color, batch=batch)
                                        #   color=(255, 255, 153), batch=batch)
                square.draw()

                label.color =(0, 0, 0, 255)
                label.draw()

    def display(self):
        print(self.board)

    def swipe(self, dir):
        # Note: 
        #   Axis == 0 -> all the values in their respective column
        #   Axis == 1 -> all the values in their respective row
        #   If we are swiping to the right (moving pieces to the right) 
        #       1. Start on the right side and loop through moving pieces to the right
        #       2. Check for combinations when moving the pieces to the right
        largest_created_val = 0

        updated = False
        updated_i = False

        self.created_val = 0
        self.created_val_count = 0
        # # Get all non-zero pieces (mask)
        # populated = self.get_nonzero()
        rev = False
        # print(dir)
        if (dir == 'R' or dir == 'D'):
            rev = True

        # Move the piece 
        # Check direction:
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
        # else:
        #     print("No moves can be made with that swipe")
        reward = reward_selection(self, updated, largest_created_val)
        
        print("Valid moves: ", self.get_valid_moves())
        # if reward < 0: reward = 0
        return (reward, self.game_over, updated)

    def get_valid_moves(self):
        action_dict_rev = {'U':0, 'R':1, 'D':2, 'L':3}

        # Create an array of invalid moves
        update_L = False
        update_R = False
        update_U = False
        update_D = False

        valid_moves = []
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
        
        if update_L:
            valid_moves.append(action_dict_rev['L'])
        if update_R:
            valid_moves.append(action_dict_rev['R'])
        if update_U:
            valid_moves.append(action_dict_rev['U'])
        if update_D:
            valid_moves.append(action_dict_rev['D'])
        
        return valid_moves
    
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
        return log_board.flatten()/np.max(log_board)

    def check_gameover(self):

        for rowcol in range(self.board_size):
                (_, got_moves, _) = self.updated_rowcol(self.board[:, rowcol], False)
                if got_moves: return
                (_, got_moves, _) = self.updated_rowcol(self.board[:, rowcol], True)
                if got_moves: return
                (_, got_moves, _) = self.updated_rowcol(self.board[rowcol, :], False)
                if got_moves: return
                (_, got_moves, _) = self.updated_rowcol(self.board[rowcol, :], True)
                if got_moves: return
        
        self.game_over = True

def test(seed, board_size, move_list):
    game = Game(seed, board_size)
    game.display()

    for move in move_list:
        game.swipe(move)
        game.display()

def play_gui(seed, board_size):
    game = Game(seed, board_size, reward_type='white_mono_corner')

play_gui(420, 4)

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
import numpy as np

def reward_selection(self, updated, largest_created_val):

        reward = 0
        # Best switchcase python can buy -- help keep things organized
        if self.reward_type == 'no_shaping':
            # Just make as many moves as possible
            if self.game_over:  # When game is over
                reward = -10    # This doesn't make sense
            elif updated :      # Everytime you make a step
                reward = 0
            else:               # Hit a wall
                reward = -1

        elif self.reward_type == 'duration_and_largest':
            # Just make as many moves as possible
            reward = 0
            if updated:
                if self.game_over:
                    reward = 10
                else:
                    if largest_created_val != 0:
                        # reward = 2*np.log2(largest_created_val)/np.max(np.log2(self.board[self.board != 0]))
                        reward = largest_created_val
                    reward += 1

        elif self.reward_type == 'combine_score':
            # Sum of all combined tiles
            if updated:
                reward = self.created_val

        elif self.reward_type == 'board_score_end_of_game':
            # Just make as many moves as possible
            reward = 0
            if updated:
                reward = self.created_val
            elif self.game_over:
                reward = self.score

        elif self.reward_type == 'end_of_game_reward':
            # Just make as many moves as possible
            reward = 0
            if self.game_over:
                reward = np.log2(np.max(self.board))

        elif self.reward_type == 'end_of_game_reward_with_proximity':
            # Just make as many moves as possible
            reward = 0
            if self.game_over:
                
                # Add the largest value around the max value
                max_loc = np.argmax(self.board)
                if (max_loc//self.board_size != 0):                 # Check if we are on the top edge
                    reward = max(reward, np.log2(self.board[(max_loc)//self.board_size-1, (max_loc)%self.board_size]))
                if (max_loc//self.board_size != self.board_size-1): # Check if we are on the bottom edge
                    reward = max(reward, np.log2(self.board[(max_loc)//self.board_size+1, (max_loc)%self.board_size]))
                if (max_loc%self.board_size != 0):                  # Check if we are on the left edge
                    reward = max(reward, np.log2(self.board[(max_loc)//self.board_size, (max_loc)%self.board_size-1]))
                if (max_loc%self.board_size != self.board_size-1):  # Check if we are on the right edge
                    reward = max(reward, np.log2(self.board[(max_loc)//self.board_size, (max_loc)%self.board_size+1]))
                    
                reward += np.log2(np.max(self.board))

        elif self.reward_type == 'end_of_game_and_duration_reward':
            # Just make as many moves as possible
            reward = 0
            if self.game_over:
                reward = np.max(self.board)
            elif updated:
                reward = 1
            else:
                reward = 0

        elif self.reward_type == 'end_of_game_duration_and_proximity_reward':
            # Just make as many moves as possible
            reward = 0
            if self.game_over:
                reward = np.max(self.board)
            elif updated:
                max_loc = np.argmax(self.board)
                if (max_loc//self.board_size != 0):                 # Check if we are on the top edge
                    reward = max(reward, self.board[(max_loc)//self.board_size-1, (max_loc)%self.board_size])/4
                if (max_loc//self.board_size != self.board_size-1): # Check if we are on the bottom edge
                    reward = max(reward, self.board[(max_loc)//self.board_size+1, (max_loc)%self.board_size])/4
                if (max_loc%self.board_size != 0):                  # Check if we are on the left edge
                    reward = max(reward, self.board[(max_loc)//self.board_size, (max_loc)%self.board_size-1])/4
                if (max_loc%self.board_size != self.board_size-1):  # Check if we are on the right edge
                    reward = max(reward, self.board[(max_loc)//self.board_size, (max_loc)%self.board_size+1])/4
                reward += largest_created_val
            else:
                reward = 0

        elif self.reward_type == 'end_of_game_duration_and_edge_reward':
            # Just make as many moves as possible
            reward = 0
            if self.game_over:
                reward = np.max(self.board)
            elif updated:
                max_loc = np.argmax(self.board)
                if (max_loc//self.board_size == 0):                 # Check if we are on the top edge
                    reward += (np.max(self.board))/4
                if (max_loc//self.board_size == self.board_size-1): # Check if we are on the bottom edge
                    reward += (np.max(self.board))/4
                if (max_loc%self.board_size == 0):                  # Check if we are on the left edge
                    reward += (np.max(self.board))/4
                if (max_loc%self.board_size == self.board_size-1):  # Check if we are on the right edge
                    reward += (np.max(self.board))/4
                reward += largest_created_val
            else:
                reward = 0

        elif self.reward_type == 'valid_move_score_reward': 
            # Just make as many moves as possible
            reward = 0
            if updated:
                reward = self.score
            else:
                reward = 0

        elif self.reward_type == 'end_of_game_score_reward': 
            # Just make as many moves as possible
            reward = 0
            if self.game_over:
                reward = self.score
            else:
                reward = 0

        elif self.reward_type == 'end_of_game_duration_and_largest_reward': # Works well!
            # Just make as many moves as possible
            reward = 0
            if self.game_over:
                reward = np.max(self.board)
            elif updated:
                reward = largest_created_val
            else:
                reward = 0
        
        elif self.reward_type == 'large_numbers':
            reward = 0
            if largest_created_val != 0:
                # reward = np.log2(largest_created_val)/np.max(np.log2(self.board[self.board != 0]))
                reward = largest_created_val
            # if reward < 0: reward = 0

        elif self.reward_type == 'duration_and_whitespace':
            reward = 0
            if largest_created_val != 0:
            # reward agent for new largest value
                if np.log2(largest_created_val)/(np.log2(np.max(self.board))) > 0:
                    reward += 1
                # reward = np.log2(largest_created_val)/(np.log2(np.max(self.board)))
                
                # reward agent for opening up spaces
                reward += max(0, len(self.get_avaliable_spaces()) - self.empty_space_val)
                self.empty_space_val = len(self.get_avaliable_spaces())
                
                # punish agent for not moving
                if not updated:
                    reward += -1
        
        else:
            raise ValueError("Invalid reward model selected")
        
        return reward
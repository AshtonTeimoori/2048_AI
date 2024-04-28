import numpy as np

def reward_selection(self, updated, largest_created_val):
        reward_vect = []
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

        elif self.reward_type == 'hs':
            # 1. The largest tile on the board
            # 2. The value of the marges made in the last move
            # 3. The number of empty cells on the board
            # 4. The number of adjacent cells that are equal
            
            reward = 0

            # 1. The largest tile on the board
            reward += np.max(self.board)

            # 2. The value of the marges made in the last move
            reward += self.created_val*50

            # 3. The number of empty cells on the board
            reward += len(self.get_avaliable_spaces())*500

            # 4. The number of adjacent cells that are equal
            adjacent_sf = 100
            for i in range(self.board_size):
                for j in range(self.board_size):
                    # Check if the cell above is equal
                    if i != 0 and self.board[i][j] == self.board[i-1][j]:
                        reward += adjacent_sf
                    # Check if the cell below is equal
                    if i != self.board_size-1 and self.board[i][j] == self.board[i+1][j]:
                        reward += adjacent_sf
                    # Check if the cell to the left is equal
                    if j != 0 and self.board[i][j] == self.board[i][j-1]:
                        reward += adjacent_sf
                    # Check if the cell to the right is equal
                    if j != self.board_size-1 and self.board[i][j] == self.board[i][j+1]:
                        reward += adjacent_sf

        elif self.reward_type == 'hs_corner':
            # 1. The largest tile on the board
            # 2. The value of the marges made in the last move
            # 3. The number of empty cells on the board
            # 4. The number of adjacent cells that are equal
            # 5. If the largest tile is on the edge of the board, double for the corner
            
            reward = 0

            # # 1. The largest tile on the board
            reward += np.max(self.board)
            reward_vect.append(np.max(self.board))
            # # 2. The value of the marges made in the last move
            # reward += self.created_val
            reward += self.created_val_count*100
            reward_vect.append(self.created_val_count*100)
            # # 3. The number of empty cells on the board
            reward += len(self.get_avaliable_spaces())*20
            reward_vect.append(len(self.get_avaliable_spaces())*20)
            # 4. The number of adjacent cells that are equal
            adjacent_sf = 100
            reward_adjacent = 0
            # for i in range(self.board_size):
            #     for j in range(self.board_size):
            #         if self.board[i][j] == 0:
            #             continue
            #         # Check if the cell above is equal
            #         if i != 0 and self.board[i][j] == self.board[i-1][j]:
            #             # reward += self.board[i][j]/3
            #             reward += adjacent_sf
            #             reward_adjacent += adjacent_sf
            #         # Check if the cell below is equal
            #         if i != self.board_size-1 and self.board[i][j] == self.board[i+1][j]:
            #             # reward += self.board[i][j]/3
            #             reward += adjacent_sf
            #             reward_adjacent += adjacent_sf
            #         # Check if the cell to the left is equal
            #         if j != 0 and self.board[i][j] == self.board[i][j-1]:
            #             # reward += self.board[i][j]/3
            #             reward += adjacent_sf
            #             reward_adjacent += adjacent_sf
            #         # Check if the cell to the right is equal
            #         if j != self.board_size-1 and self.board[i][j] == self.board[i][j+1]:
            #             # reward += self.board[i][j]/3
            #             reward += adjacent_sf
            #             reward_adjacent += adjacent_sf
            
            reward_vect.append(reward_adjacent)
            
            # 5. If the largest tile is on the edge of the board, double for the corner
            # side_sf = 50
            side_sf = np.max(self.board)
            max_loc = np.argmax(self.board)
            reward_side = 0
            if (max_loc//self.board_size == 0):                 # Check if we are on the top edge
                # reward += np.log2(np.max(self.board))*side_sf
                reward += side_sf
                reward_side += side_sf
            if (max_loc//self.board_size == self.board_size-1): # Check if we are on the bottom edge
                # reward += np.log2(np.max(self.board))*side_sf
                reward += side_sf
                reward_side += side_sf
            if (max_loc%self.board_size == 0):                  # Check if we are on the left edge
                # reward += np.log2(np.max(self.board))*side_sf
                reward += side_sf
                reward_side += side_sf
            if (max_loc%self.board_size == self.board_size-1):  # Check if we are on the right edge
                # reward += np.log2(np.max(self.board))*side_sf
                reward += side_sf
                reward_side += side_sf
            reward_vect.append(reward_side)

            # 6. The number of adjacent cells that are one away
            adjacent_sf = 20
            reward_adjacent = 0
            for i in range(self.board_size):
                for j in range(self.board_size):
                    # Check if the cell above is equal
                    if i != 0 and self.board[i][j]/2 == self.board[i-1][j]:
                        # reward += self.board[i-1][j]/6
                        reward += adjacent_sf
                        reward_adjacent += adjacent_sf
                    # Check if the cell below is equal
                    if i != self.board_size-1 and self.board[i][j]/2 == self.board[i+1][j]:
                        # reward += self.board[i+1][j]/6
                        reward += adjacent_sf
                        reward_adjacent += adjacent_sf
                    # Check if the cell to the left is equal
                    if j != 0 and self.board[i][j]/2 == self.board[i][j-1]:
                        # reward += self.board[i][j-1]/6
                        reward += adjacent_sf
                        reward_adjacent += adjacent_sf
                    # Check if the cell to the right is equal
                    if j != self.board_size-1 and self.board[i][j]/2 == self.board[i][j+1]:
                        # reward += self.board[i][j+1]/6
                        reward += adjacent_sf
                        reward_adjacent += adjacent_sf
            reward_vect.append(reward_adjacent)
            self.reward_vect = reward_vect  
            
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
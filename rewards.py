import numpy as np

def reward_selection(self, updated, largest_created_val=0):

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
            reward += self.created_val

            # 3. The number of empty cells on the board
            reward += len(self.get_avaliable_spaces())*50

            # 4. The number of adjacent cells that are equal
            adjacent_sf = 100
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board[i][j] == 0:
                        continue
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

        elif self.reward_type == 'white_mono_corner':
            # 1. The number of empty cells on the board
            # 2. The number of monotonically increasing rows and columns (>=)
            #    - 32->16, since this is 1 off, multiply by 1/1. 32->8 is 2 off, multiply by 1/2
            # 3. The largest tile on the board is on the edge or in a corner 

            reward = 0
            reward_vect = []
            # 1. The number of empty cells on the board
            white_space_sf = 1
            reward += len(self.get_avaliable_spaces())*white_space_sf
            reward_vect.append(len(self.get_avaliable_spaces())*white_space_sf)
            # print("White space: ", reward)

            # 2. The number of monotonically increasing rows and columns (>=)
            mono_sf = 10

            # Check rows
            row_reward = 0.0
            
            for i in range(self.board_size):
                temp_row = 0.0
                temp_val = 0.0
                same_tile_reward = 0.0
                for j in range(self.board_size):
                    if self.board[i][j] == 0:
                        continue
                    # Grab first non-zero value
                    if temp_val == 0:
                        temp_val = self.board[i][j]
                        continue
                    if np.log2(temp_val) == np.log2(self.board[i][j]):
                        same_tile_reward += 1.5
                        # same_tile_reward += temp_val
                    else:
                        temp_row += (np.log2(self.board[i][j]) - np.log2(temp_val))
                    temp_val = self.board[i][j]
                row_reward += abs(temp_row) + same_tile_reward

            # Check columns
            col_reward = 0.0
            for j in range(self.board_size):
                temp_col = 0.0
                temp_val = 0.0
                same_tile_reward = 0.0
                for i in range(self.board_size):
                    if self.board[i][j] == 0:
                        continue
                    # Grab first non-zero value
                    if temp_val == 0:
                        temp_val = self.board[i][j]
                        continue
                    if np.log2(temp_val) == np.log2(self.board[i][j]):
                        same_tile_reward += 1.5
                        # same_tile_reward += temp_val
                    else:
                        temp_col += (np.log2(self.board[i][j]) - np.log2(temp_val))
                    temp_val = self.board[i][j]
                col_reward += abs(temp_col) + same_tile_reward

            reward += mono_sf*(row_reward + col_reward)# + same_tile_reward
            reward_vect.append(mono_sf*(row_reward + col_reward))# + same_tile_reward)
            # print("Mono-row: ", mono_sf*(row_reward))
            # print("Mono-col: ", mono_sf*(col_reward))

            # 3. The largest tile on the board is on the edge or in a corner
            side_sf = 150
            r3=0
            r6=0
            r7=0
            max_locs = np.argwhere(self.board == np.amax(self.board))
            for max_loc in max_locs:
                if (max_loc[0] == 0 and max_loc[1] == 0): # TL Corner
                    r3 = 2*side_sf + np.max(self.board)
                    r6 += self.board[max_loc[0]+1, max_loc[1]]
                    r6 += self.board[max_loc[0], max_loc[1]+1]
                    r6 += self.board[max_loc[0]+1, max_loc[1]+1]
                    r7=0 
                    break
                elif (max_loc[0] == 0 and max_loc[1] == self.board_size-1): # TR Corner
                    r3 = 2*side_sf + np.max(self.board)
                    r6 += self.board[max_loc[0]+1, max_loc[1]]
                    r6 += self.board[max_loc[0], max_loc[1]-1]
                    r6 += self.board[max_loc[0]+1, max_loc[1]-1]
                    r7=0
                    break
                elif (max_loc[0] == self.board_size-1 and max_loc[1] == 0): # BL Corner
                    r3 = 2*side_sf + np.max(self.board)
                    r6 += self.board[max_loc[0]-1, max_loc[1]]
                    r6 += self.board[max_loc[0], max_loc[1]+1]
                    r6 += self.board[max_loc[0]-1, max_loc[1]+1]
                    r7=0
                    break
                elif (max_loc[0] == self.board_size-1 and max_loc[1] == self.board_size-1): # BR Corner
                    r3 = 2*side_sf + np.max(self.board)
                    r6 += self.board[max_loc[0]-1, max_loc[1]]
                    r6 += self.board[max_loc[0], max_loc[1]-1]
                    r6 += self.board[max_loc[0]-1, max_loc[1]-1]
                    r7=0
                    break
                else:
                    # reward model for non-max out from the corner
                    
                    corner_indices = [[0,0], [0,self.board_size-1], [self.board_size-1,0], [self.board_size-1,self.board_size-1]]
                    for corners in corner_indices:
                        x = corners[0]
                        y = corners[1]
                        if corners == [0,0]:
                            r7 += (self.board[x,y] - self.board[x+1,y+1] - self.board[x+1,y] - self.board[x,y+1])
                        elif corners == [0,self.board_size-1]:
                            r7 += (self.board[x,y] - self.board[x+1,y-1] - self.board[x+1,y] - self.board[x,y-1])
                        elif corners == [self.board_size-1,0]:
                            r7 += (self.board[x,y] - self.board[x-1,y+1] - self.board[x-1,y] - self.board[x,y+1])
                        elif corners == [self.board_size-1,self.board_size-1]:
                            r7 += (self.board[x,y] - self.board[x-1,y-1] - self.board[x-1,y] - self.board[x,y-1])
                    
                    
                    if (max_loc[0] == 0):                 # Check if we are on the top edge
                        # r3 += np.log2(np.max(self.board))*side_sf
                        r3 = side_sf
                    elif (max_loc[0] == self.board_size-1): # Check if we are on the bottom edge
                        # r3 += np.log2(np.max(self.board))*side_sf
                        r3 = side_sf
                    elif (max_loc[1] == 0):                  # Check if we are on the left edge
                        # r3 += np.log2(np.max(self.board))*side_sf
                        r3 = side_sf
                    elif (max_loc[1] == self.board_size-1):  # Check if we are on the right edge
                        # r3 += np.log2(np.max(self.board))*side_sf
                        r3 = side_sf
            reward += r3
            
            reward_vect.append(r3)
            # print("Corner: ", r3)

            # 4. Created max tile
            r4 = 0
            if largest_created_val == np.max(np.max(self.board)):
                r4 = 120
            # print("Making max tile: ", r4)
            reward += r4
            reward_vect.append(r4)
            # 5. Combined 
            r5=self.created_val_count*40
            reward += r5
            reward_vect.append(r5)
            
            # reward for sum of cells around corner max tile
            reward += r6
            reward_vect.append(r6)
            
            # reward for moving smaller tile away from corner
            reward += r7
            reward_vect.append(r7)
            
            self.reward_vect = reward_vect  

        elif self.reward_type == 'hs_corner':
            # 1. The largest tile on the board
            # 2. The value of the marges made in the last move
            # 3. The number of empty cells on the board
            # 4. The number of adjacent cells that are equal
            # 5. If the largest tile is on the edge of the board, double for the corner
            
            reward = 0

            # # 1. The largest tile on the board
            # reward += np.max(self.board)

            # # 2. The value of the marges made in the last move
            # reward += self.created_val
            r1=self.created_val_count*100
            reward += r1
            # print("#2", reward)
            # # 3. The number of empty cells on the board
            # reward += len(self.get_avaliable_spaces())*50

            # 4. The number of adjacent cells that are equal
            adjacent_sf = 50
            r4=0
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board[i][j] == 0:
                        continue
                    # Check if the cell above is equal
                    if i != 0 and self.board[i][j] == self.board[i-1][j]:
                        # reward += self.board[i][j]/3
                        r4 += adjacent_sf
                    # Check if the cell below is equal
                    if i != self.board_size-1 and self.board[i][j] == self.board[i+1][j]:
                        # r4 += self.board[i][j]/3
                        r4 += adjacent_sf
                    # Check if the cell to the left is equal
                    if j != 0 and self.board[i][j] == self.board[i][j-1]:
                        # r4 += self.board[i][j]/3
                        r4 += adjacent_sf
                    # Check if the cell to the right is equal
                    if j != self.board_size-1 and self.board[i][j] == self.board[i][j+1]:
                        # r4 += self.board[i][j]/3
                        r4 += adjacent_sf
            reward += r4
            # print("#4: ", r4)

            # 5. If the largest tile is on the edge of the board, double for the corner
            # side_sf = 50
            side_sf = 500
            r5=0
            max_locs = np.argwhere(self.board == np.amax(self.board))
            for max_loc in max_locs:
                # r5=0
                if (max_loc[0] == 0 and max_loc[1] == 0): # TL Corner
                    r5 = 2*side_sf
                    break
                elif (max_loc[0] == 0 and max_loc[1] == self.board_size-1): # TR Corner
                    r5 = 2*side_sf
                    break
                elif (max_loc[0] == self.board_size-1 and max_loc[1] == 0): # BL Corner
                    r5 = 2*side_sf
                    break
                elif (max_loc[0] == self.board_size-1 and max_loc[1] == self.board_size-1): # BR Corner
                    r5 = 2*side_sf
                    break
                elif (max_loc[0] == 0):                 # Check if we are on the top edge
                    # r5 += np.log2(np.max(self.board))*side_sf
                    r5 = side_sf
                elif (max_loc[0] == self.board_size-1): # Check if we are on the bottom edge
                    # r5 += np.log2(np.max(self.board))*side_sf
                    r5 = side_sf
                elif (max_loc[1] == 0):                  # Check if we are on the left edge
                    # r5 += np.log2(np.max(self.board))*side_sf
                    r5 = side_sf
                elif (max_loc[1] == self.board_size-1):  # Check if we are on the right edge
                    # r5 += np.log2(np.max(self.board))*side_sf
                    r5 = side_sf
            reward += r5
            # print("#5: ", r5)

            # 6. The number of adjacent cells that are one away
            adjacent_sf = 20
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board[i][j] == 0:
                        continue
                    # Check if the cell above is equal
                    if i != 0 and self.board[i][j]/2 == self.board[i-1][j]:
                        # reward += self.board[i-1][j]/6
                        reward += adjacent_sf
                    # Check if the cell below is equal
                    if i != self.board_size-1 and self.board[i][j]/2 == self.board[i+1][j]:
                        # reward += self.board[i+1][j]/6
                        reward += adjacent_sf
                    # Check if the cell to the left is equal
                    if j != 0 and self.board[i][j]/2 == self.board[i][j-1]:
                        # reward += self.board[i][j-1]/6
                        reward += adjacent_sf
                    # Check if the cell to the right is equal
                    if j != self.board_size-1 and self.board[i][j]/2 == self.board[i][j+1]:
                        # reward += self.board[i][j+1]/6
                        reward += adjacent_sf
                        
            # 7. Created max tile
            r6 = 0
            if largest_created_val == np.max(np.max(self.board)):
                r6 = 500
            # print("#6: ", r6)
            reward += r6
        
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
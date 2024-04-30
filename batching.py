
BOARDBUFF = namedtuple("BOARDBUFF", ["S", "D"]) # Board state, Game duration

class BoardBuffer(object):
    def __init__(self, size):
        self.buffer = deque([], size)

    def push(self, *args):
        self.buffer.append(BOARDBUFF(*args))

    def pop_sample(self):
        pop_index = random.randint(0, len(self.buffer)-1)
        return_board = self.buffer[pop_index]
        del self.buffer[pop_index]
        return return_board
    
    def __len__(self):
        return len(self.buffer)
    
BOARDCACHE = namedtuple("BOARDCACHE", ["S", "D"]) # Board state, Game duration

class BoardCache(object):
    def __init__(self, size):
        self.buffer = deque([], size)

    def clear(self):
        self.buffer.clear()

    def push(self, *args):
        self.buffer.append(BOARDCACHE(*args))

    def get_first(self):
        return self.buffer.popleft()
    
    def __len__(self):
        return len(self.buffer)



episodic_rewards = []
episodic_loss = []
episodic_epsilon = []
episodic_duration = []
episdoic_max_tile = []
episodic_invalid_moves_made_count = []

def DQN_network(episodes):
    
    # burn in to collect batch data
    for i in range(1000):
        if (not XTRA_IN):
            S = env.reset()   
        else:
            S = np.append(env.reset(), [env.game_duration, env.largest_value])
        
        if (OHE):
            S = encode_state(S).flatten()
            S = torch.tensor(S, dtype=torch.float32, device=device)
        else:
            S = torch.tensor([S], dtype=torch.float32, device=device)
            
            while not terminated:
                valid_actions = env.get_possible_moves()
                A = random.randchoice(valid_actions)
                (reward, terminated, updated, invalid_moves, invalid_moves_made) = env.swipe(action_dict[A])
                if (not XTRA_IN):
                    S_prime = env.get_flat_board()
                else:
                    S_prime = np.append(env.get_flat_board(), [env.game_duration, env.largest_value])
                    
                if (OHE):
                    S_prime = encode_state(S_prime).flatten()
                    S_prime = [0] if terminated else tensor(S_prime, dtype=torch.float32, device=device)
                else:
                    # S_prime = None if terminated else tensor(torch.FloatTensor(S_prime).to(device), requires_grad=True)
                    S_prime = [0] if terminated else tensor([S_prime], dtype=torch.float32, device=device)

                # Store the transition
                # RB.push(S, A, tensor([[reward]], dtype=torch.float32, device=device), 
                #         S_prime, tensor(not terminated, device=device, dtype=torch.bool))
                RB.push(S, A, tensor([[reward]], dtype=torch.float32, device=device), 
                        S_prime, not terminated)

                S = S_prime
    
    
    global max_game
    save_tag = "hs"
    start_time = time.time()
    T = 0
    for epi in range(episodes):
        
        # Play 50 games in parallel
        for _ in range(50):
            if (not XTRA_IN):
                S = env.reset()   
            else:
                S = np.append(env.reset(), [env.game_duration, env.largest_value])

            if (OHE):
                S = encode_state(S).flatten()
                S = torch.tensor(S, dtype=torch.float32, device=device)
            else:
                S = torch.tensor([S], dtype=torch.float32, device=device)

            episodic_reward = 0
            episodic_mean_loss = 0
            terminated = False
            episodic_invalid_moves_made = 0

            mini_duration = 0

            invalid_moves = []
            reward_vect = []
            game_states = []
            action_vect = []
            while not terminated:
                T += 1
                mini_duration += 1

                epsilon = getEpsilon(env.game_duration)
                # Choose action
                A = epsilonGreedy(S, policy_net, nA, epsilon, invalid_moves)
                # Take step
                (reward, terminated, updated, invalid_moves, invalid_moves_made) = env.swipe(action_dict[A])
                if (not XTRA_IN):
                    S_prime = env.get_flat_board()
                else:
                    S_prime = np.append(env.get_flat_board(), [env.game_duration, env.largest_value])

                episodic_invalid_moves_made += invalid_moves_made

                if (OHE):
                    S_prime = encode_state(S_prime).flatten()
                    S_prime = [0] if terminated else tensor(S_prime, dtype=torch.float32, device=device)
                else:
                    # S_prime = None if terminated else tensor(torch.FloatTensor(S_prime).to(device), requires_grad=True)
                    S_prime = [0] if terminated else tensor([S_prime], dtype=torch.float32, device=device)

                # Store the transition
                # RB.push(S, A, tensor([[reward]], dtype=torch.float32, device=device), 
                #         S_prime, tensor(not terminated, device=device, dtype=torch.bool))
                RB.push(S, A, tensor([[reward]], dtype=torch.float32, device=device), 
                        S_prime, not terminated)

                S = S_prime

                # Update the networks networks
                if len(RB) > BATCH_SIZE:
                    episodic_mean_loss += train()

                episodic_reward += reward

                if T%20==0:
                    # Soft update of the target network's weights
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    target_net.load_state_dict(target_net_state_dict)

                game_states.append(np.copy(env.board))
                action_vect.append(action_dict[A])
                reward_vect.append(env.reward_vect)

            episodic_epsilon.append(epsilon)
            episodic_loss.append(episodic_mean_loss/T)
            episodic_rewards.append(episodic_reward)
            episodic_duration.append(env.game_duration)
            episdoic_max_tile.append(np.log2(env.largest_value))
            episodic_invalid_moves_made_count.append(episodic_invalid_moves_made)

            # if len(episodic_duration) < 10:
            #     max_game = 0
            if len(episodic_duration) < 50:
                max_game = max(max_game, env.game_duration)
            else:
                max_game = max(episodic_duration[-50:])
            if epi % 100 == 0:
                save_string = "_policy_weights_episode_"+str(epi).zfill(4)
                torch.save(target_net.state_dict(), "./trained_models/"+save_tag+"_"+"target"+save_string+".pth")
                torch.save(policy_net.state_dict(), "./trained_models/"+save_tag+"_"+"policy"+save_string+".pth")
                plot_multi(["Reward History", "Loss History", "Duration", "Max Tile History", "Epsilon History"], 
                           ["Reward", "Loss", "Duration", "Max Tile Value", "Epsilon"], 
                           [episodic_rewards, episodic_loss, episodic_duration, episdoic_max_tile, episodic_epsilon], save_string="model_"+save_tag+"")

            if epi % 5 == 0:
                print(epi)
                env.display()
                plot_multi(["Training Rewards...", "Training Loss...", "Duration...", "Max Tile Value...", "Epsilon Value..."], 
                            ["Reward", "Mean Episode Loss", "Duration", "Max Tile Value", "Epsilon"], 
                            [episodic_rewards, episodic_loss, episodic_duration, episdoic_max_tile, episodic_epsilon])

            if epi % 25 == 0:
                with open('rewards.txt', 'w') as f:
                    for i, state in enumerate(game_states):
                        f.write('State:\n {}\n'.format(state))
                        f.write('Action: {}\n'.format(action_vect[i]))
                        f.write('Reward: {}\n'.format(reward_vect[i]))
                        f.write('Total Reward: {}\n'.format(sum(reward_vect[i])))
                        f.write('\n')


        if (not XTRA_IN):
            S = env.reset()   
        else:
            S = np.append(env.reset(), [env.game_duration, env.largest_value])
        
        if (OHE):
            S = encode_state(S).flatten()
            S = torch.tensor(S, dtype=torch.float32, device=device)
        else:
            S = torch.tensor([S], dtype=torch.float32, device=device)

        episodic_reward = 0
        episodic_mean_loss = 0
        terminated = False
        episodic_invalid_moves_made = 0
        
        mini_duration = 0

        invalid_moves = []
        reward_vect = []
        game_states = []
        action_vect = []
        while not terminated:
            T += 1
            mini_duration += 1

            epsilon = getEpsilon(env.game_duration)
            # Choose action
            A = epsilonGreedy(S, policy_net, nA, epsilon, invalid_moves)
            # Take step
            (reward, terminated, updated, invalid_moves, invalid_moves_made) = env.swipe(action_dict[A])
            if (not XTRA_IN):
                S_prime = env.get_flat_board()
            else:
                S_prime = np.append(env.get_flat_board(), [env.game_duration, env.largest_value])
            
            episodic_invalid_moves_made += invalid_moves_made

            if (OHE):
                S_prime = encode_state(S_prime).flatten()
                S_prime = [0] if terminated else tensor(S_prime, dtype=torch.float32, device=device)
            else:
                # S_prime = None if terminated else tensor(torch.FloatTensor(S_prime).to(device), requires_grad=True)
                S_prime = [0] if terminated else tensor([S_prime], dtype=torch.float32, device=device)

            # Store the transition
            # RB.push(S, A, tensor([[reward]], dtype=torch.float32, device=device), 
            #         S_prime, tensor(not terminated, device=device, dtype=torch.bool))
            RB.push(S, A, tensor([[reward]], dtype=torch.float32, device=device), 
                    S_prime, not terminated)

            S = S_prime
            
            # Update the networks networks
            if len(RB) > BATCH_SIZE:
                episodic_mean_loss += train()
                
            episodic_reward += reward

            if T%20==0:
                # Soft update of the target network's weights
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)
            
            game_states.append(np.copy(env.board))
            action_vect.append(action_dict[A])
            reward_vect.append(env.reward_vect)

        episodic_epsilon.append(epsilon)
        episodic_loss.append(episodic_mean_loss/T)
        episodic_rewards.append(episodic_reward)
        episodic_duration.append(env.game_duration)
        episdoic_max_tile.append(np.log2(env.largest_value))
        episodic_invalid_moves_made_count.append(episodic_invalid_moves_made)

        # if len(episodic_duration) < 10:
        #     max_game = 0
        if len(episodic_duration) < 50:
            max_game = max(max_game, env.game_duration)
        else:
            max_game = max(episodic_duration[-50:])
        if epi % 100 == 0:
            save_string = "_policy_weights_episode_"+str(epi).zfill(4)
            torch.save(target_net.state_dict(), "./trained_models/"+save_tag+"_"+"target"+save_string+".pth")
            torch.save(policy_net.state_dict(), "./trained_models/"+save_tag+"_"+"policy"+save_string+".pth")
            plot_multi(["Reward History", "Loss History", "Duration", "Max Tile History", "Epsilon History"], 
                       ["Reward", "Loss", "Duration", "Max Tile Value", "Epsilon"], 
                       [episodic_rewards, episodic_loss, episodic_duration, episdoic_max_tile, episodic_epsilon], save_string="model_"+save_tag+"")

        if epi % 5 == 0:
            print(epi)
            env.display()
            plot_multi(["Training Rewards...", "Training Loss...", "Duration...", "Max Tile Value...", "Epsilon Value..."], 
                        ["Reward", "Mean Episode Loss", "Duration", "Max Tile Value", "Epsilon"], 
                        [episodic_rewards, episodic_loss, episodic_duration, episdoic_max_tile, episodic_epsilon])
            
        if epi % 25 == 0:
            with open('rewards.txt', 'w') as f:
                for i, state in enumerate(game_states):
                    f.write('State:\n {}\n'.format(state))
                    f.write('Action: {}\n'.format(action_vect[i]))
                    f.write('Reward: {}\n'.format(reward_vect[i]))
                    f.write('Total Reward: {}\n'.format(sum(reward_vect[i])))
                    f.write('\n')
                
    
    delta_time = time.time()-start_time
    
    plt.ioff()
    plt.show()
    
    # Save data
    # data_file = open("./trainged_models/data.json", 'w+')
    # json_data = {"episodic_rewards": episodic_rewards, 
    #                 "episodic_loss": episodic_loss, 
    #                 "episodic_epsilon": episodic_epsilon,
    #                 "training_time": delta_time
    #                 }
    # json.dump(json_data, data_file)
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from cube import Cube
import kociemba as koc
import time  # Add time module for timing
import os
import argparse
import json  # Add import for JSON parsing
import subprocess  # For calling gen.py
import sys

# Create modelCheckpoints directory if it doesn't exist
os.makedirs("modelCheckpoints", exist_ok=True)

# Define possible moves
MOVES = ["U", "U'", "U2", "D", "D'", "D2", "L", "L'", "L2", "R", "R'", "R2", "F", "F'", "F2", "B", "B'", "B2"]

class CubeEnvironment:
    def __init__(self, max_steps=25, scramble_moves=1, use_pregenerated=False):
        self.cube = Cube()
        self.max_steps = max_steps  # Solution shouldn't be more than 25 moves
        self.current_step = 0
        self.agent_moves = []  # Track the sequence of moves
        self.scramble_moves = scramble_moves  # Number of scramble moves
        self.use_pregenerated = use_pregenerated  # Whether to use pregenerated scrambles
        self.scramble_usage_count = 0  # Track how many scrambles have been used
        
        # Define move cancellation patterns
        self.opposite_faces = {
            'U': 'D', 'D': 'U',
            'F': 'B', 'B': 'F',
            'R': 'L', 'L': 'R'
        }
        
        # Load pregenerated scrambles if required
        if self.use_pregenerated:
            self.load_scrambles(scramble_moves)
            
    def load_scrambles(self, num_moves):
        """Load pregenerated scrambles from file"""
        self.pregenerated_scrambles = []
        
        # Create scrambles directory if it doesn't exist
        os.makedirs("scrambles", exist_ok=True)
        
        # Define path to scramble file
        filepath = os.path.join("scrambles", f"{num_moves}movescramble.txt")
        
        # Check if scramble file exists, generate if it doesn't
        if not os.path.exists(filepath) and self.use_pregenerated:
            print(f"Scramble file {filepath} not found. Generating now...")
            try:
                subprocess.run(["python", "gen.py", "--level", str(num_moves)], check=True)
                if not os.path.exists(filepath):
                    print(f"Failed to generate scramble file. Will use random scrambles instead.")
                    self.use_pregenerated = False
                    return
            except Exception as e:
                print(f"Error generating scrambles: {e}")
                self.use_pregenerated = False
                return
        
        # Load scrambles from file  
        try:
            if not os.path.exists(filepath):
                print(f"Warning: Scramble file {filepath} not found. Will use random scrambles instead.")
                self.use_pregenerated = False
                return
                
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        scramble_data = json.loads(line.strip())
                        self.pregenerated_scrambles.append(scramble_data)
                    except json.JSONDecodeError:
                        continue
                        
            print(f"Loaded {len(self.pregenerated_scrambles)} pregenerated scrambles with {num_moves} solution length")
            if not self.pregenerated_scrambles:
                self.use_pregenerated = False
                print("No valid scrambles found, switching to random scrambles")
        except Exception as e:
            print(f"Error loading scrambles: {e}")
            self.use_pregenerated = False
    
    def check_and_regenerate_scrambles(self):
        """Check if we're running out of scrambles and regenerate if needed"""
        # If we've used a significant portion of scrambles, regenerate
        if self.use_pregenerated and self.pregenerated_scrambles:
            self.scramble_usage_count += 1
            
            # Regenerate when we've used all scrambles (typically 50,000)
            if self.scramble_usage_count >= len(self.pregenerated_scrambles):
                print(f"\nUsed all {len(self.pregenerated_scrambles)} scrambles. Generating a new set...")
                try:
                    # Generate new scrambles
                    subprocess.run(["python", "gen.py", "--level", str(self.scramble_moves)], check=True)
                    # Reset counter
                    self.scramble_usage_count = 0
                    # Reload scrambles
                    self.load_scrambles(self.scramble_moves)
                    print(f"Successfully regenerated scrambles for level {self.scramble_moves}")
                except Exception as e:
                    print(f"Error regenerating scrambles: {e}")
        
    def reset(self):
        self.cube = Cube()
        self.current_step = 0
        self.agent_moves = []  # Reset the moves list
        
        # Check if we need to regenerate scrambles
        self.check_and_regenerate_scrambles()
        
        scramble_moves = []
        if self.use_pregenerated and self.pregenerated_scrambles:
            # Use a pregenerated scramble
            scramble_data = random.choice(self.pregenerated_scrambles)
            scramble = scramble_data["scramble"]
            self.cube.apply_algorithm(scramble)
            # Store the scramble moves
            scramble_moves = scramble.split()
        else:
            # Apply random scramble moves as before
            for _ in range(self.scramble_moves):
                move = random.choice(MOVES)
                scramble_moves.append(move)
                self.cube.apply_algorithm(move)
        
        # Store the scramble moves (these are not agent moves, but we'll use this to track the scramble)
        self.scramble_sequence = scramble_moves
                
        return self._get_state()
    
    def step(self, action):
        # Apply the move corresponding to the action
        self.cube.apply_algorithm(MOVES[action])
        self.agent_moves.append(MOVES[action])  # Record the move
        self.current_step += 1
        
        # Check if cube is solved
        done = self.cube.is_solved()
        
        # Calculate reward
        reward = self._calculate_reward(done, action)
        
        # Check if maximum steps reached
        if self.current_step >= self.max_steps:
            done = True
        
        return self._get_state(), reward, done
    
    def _get_state(self):
        # Convert cube state to a flat array
        state = []
        for face in self.cube.faces:
            for sticker in face:
                # One-hot encode the colors
                if sticker == 'W': state.extend([1, 0, 0, 0, 0, 0])
                elif sticker == 'O': state.extend([0, 1, 0, 0, 0, 0])
                elif sticker == 'G': state.extend([0, 0, 1, 0, 0, 0])
                elif sticker == 'R': state.extend([0, 0, 0, 1, 0, 0])
                elif sticker == 'B': state.extend([0, 0, 0, 0, 1, 0])
                elif sticker == 'Y': state.extend([0, 0, 0, 0, 0, 1])
        return np.array(state, dtype=np.float32)
    
    def _calculate_reward(self, solved, action):
        reward = 0
        
        # --- MAJOR OUTCOME REWARDS ---
        # Higher base reward for solving the cube
        if solved:
            reward += 200  # Increased from 100 for stronger positive signal
        else:
            reward -= 5  # Reduced negative reward for not solving yet - was -10 before
            
        # --- MOVE EFFICIENCY PENALTIES ---
        # Collect penalties instead of directly applying them
        penalties = []
        
        # Penalize repeating moves (like R R R instead of R')
        if len(self.agent_moves) >= 3:
            last_three_moves = self.agent_moves[-3:]
            if last_three_moves[0] == last_three_moves[1] == last_three_moves[2]:
                penalties.append(-15)  # Increased penalty for three consecutive identical moves
        
        # Penalize sequences like R2 R or R2 R' (inefficient move combinations)
        if len(self.agent_moves) >= 2:
            last_move = self.agent_moves[-1]
            prev_move = self.agent_moves[-2]
            
            # Check if previous move was a double move (R2) and current move is on the same face (R or R')
            if "2" in prev_move and not "2" in last_move:
                # If they're moves on the same face
                if prev_move.replace("2", "") == last_move.replace("'", ""):
                    penalties.append(-15)  # Increased penalty for inefficient sequence
            
            # Check for the reverse case: R followed by R2 (another inefficient sequence)
            elif "2" in last_move and not "2" in prev_move:
                # If they're moves on the same face
                if last_move.replace("2", "") == prev_move.replace("'", ""):
                    penalties.append(-15)  # Increased penalty for inefficient sequence
        
        # Penalize inverse moves that cancel each other (like R R')
        if len(self.agent_moves) >= 2:
            last_move = self.agent_moves[-1]
            prev_move = self.agent_moves[-2]
            
            # Check for move cancellations (simplified check)
            # For basic moves like R and R'
            if (last_move.replace("'", "") == prev_move.replace("'", "") and 
                ("'" in last_move) != ("'" in prev_move) and 
                "2" not in last_move and "2" not in prev_move):
                penalties.append(-20)  # Increased penalty for immediately canceling moves
        
        # Penalize inefficient sequences like B2 F B2 (where B2 B2 would cancel out)
        if len(self.agent_moves) >= 3:
            last_move = self.agent_moves[-1]
            second_last_move = self.agent_moves[-2]
            third_last_move = self.agent_moves[-3]
            
            # Check for patterns like X2 Y X2 where X2 cancels out
            if "2" in last_move and "2" in third_last_move:
                # If they're the same face (like U2 F U2)
                if last_move.replace("2", "") == third_last_move.replace("2", ""):
                    penalties.append(-15)  # Increased penalty for inefficient sequences
            
            # Check for sequences where moves on opposite faces interact inefficiently
            # For example B2 F B2 is just F
            last_face = last_move[0]  # Get the face of the last move
            third_last_face = third_last_move[0]  # Get the face of the third-last move
            
            # If the moves are on opposite faces and both are double moves (2)
            if "2" in last_move and "2" in third_last_move:
                if self.opposite_faces.get(last_face) == third_last_face:
                    penalties.append(-15)  # Increased penalty for inefficient sequences
            
            # Check for patterns like L R L' where a move is followed by another move
            # and then the inverse of the first move
            if ("2" not in third_last_move and 
                third_last_move.replace("'", "") != second_last_move.replace("'", "") and 
                third_last_move.replace("'", "") == last_move.replace("'", "") and
                ("'" in third_last_move) != ("'" in last_move)):
                penalties.append(-15)  # Increased penalty for inefficient sequences like L R L'
        
        # Apply only the worst penalty instead of stacking them all
        if penalties:
            reward += min(penalties)
        
        # --- USE KOCIEMBA FOR IMPROVED REWARD SIGNALS ---
        try:
            # Get current cube state in kociemba format
            current_state = self.cube.to_kociemba_string()
            solution = koc.solve(current_state)
            solution_length = len(solution.split())
            
            # --- PROGRESS-BASED REWARDS ---
            # Count number of solved stickers per face
            solved_stickers = 0
            for face in self.cube.faces:
                center_color = face[4]  # Center sticker color
                for sticker in face:
                    if sticker == center_color:
                        solved_stickers += 1
            
            # Normalized sticker progress (54 stickers total, 6 centers always solved)
            # 48 stickers can change, 9 correct stickers per face would be perfect
            sticker_progress = (solved_stickers - 6) / 48.0  # Range 0.0 to 1.0
            
            # Add small reward based on solved stickers (boost signal for incremental progress)
            reward += sticker_progress * 5
            
            # --- SOLUTION DISTANCE REWARD/PENALTY ---
            # Stronger but still gentle scaling for solution distance
            # Shorter solution = less negative reward (better)
            reward -= 0.8 * solution_length
            
            # --- SOLUTION QUALITY REWARDS (when solved) ---
            agent_solution_length = len(self.agent_moves)
            if solved:
                # Compare agent's solution with Kociemba's optimal solution
                if agent_solution_length < solution_length:
                    # More reward for solving in fewer moves than Kociemba
                    reward += (solution_length - agent_solution_length) * 10
                elif agent_solution_length > solution_length + 5:
                    # Only penalize if significantly longer than Kociemba (5+ moves more)
                    reward -= (agent_solution_length - solution_length - 5) * 1.5
                
                # Stronger scaled reward for solutions under 20 moves (God's number)
                if agent_solution_length <= 20:
                    reward += 10 * (20 - agent_solution_length)  # Higher reward for shorter solutions
            
            # --- REWARD FOR IMPROVEMENT IN SOLUTION LENGTH ---
            # Store the current solution length for this specific cube state
            # So we can compare if the next move improves it
            self.current_solution_length = solution_length
        except Exception as e:
            # If kociemba fails (invalid cube state), give a small negative reward
            reward -= 2  # Slightly more negative to discourage invalid states
        
        return reward

class DQN(nn.Module):
    def __init__(self, state_size, action_size, dueling=True):
        super(DQN, self).__init__()
        # Common feature layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Dueling architecture - separate value and advantage streams
        self.dueling = dueling
        if dueling:
            # Value stream - estimates state value V(s)
            self.value_stream = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)  # Single value output
            )
            
            # Advantage stream - estimates advantages A(s,a)
            self.advantage_stream = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)  # One output per action
            )
        else:
            # Traditional Q-value output
            self.q_output = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)
            )
        
    def forward(self, x):
        features = self.feature_layers(x)
        
        if self.dueling:
            # Compute value and advantage separately
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            # Combine value and advantage for Q-values
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values
        else:
            # Traditional DQN
            return self.q_output(features)

class DQNAgent:
    def __init__(self, state_size, action_size, 
                 learning_rate=0.001,             # Higher learning rate for faster learning
                 memory_size=100000,              # Larger memory for more experiences
                 gamma=0.99,                      # Discount factor for future rewards
                 epsilon_start=1.0,               # Starting exploration rate
                 epsilon_min=0.05,                # Minimum exploration rate
                 epsilon_decay=0.995,             # Faster epsilon decay
                 target_update_freq=1000,         # How often to update target network
                 double_dqn=True,                 # Use Double DQN 
                 dueling_dqn=True,                # Use Dueling DQN architecture
                 prioritized_replay=True,         # Use prioritized experience replay
                 alpha=0.6,                       # Priority exponent (0 = uniform sampling)
                 beta=0.4,                        # Initial importance sampling weight
                 beta_increment=0.001):           # How much to increase beta each sampling
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.update_counter = 0
        
        # Prioritized experience replay parameters
        self.prioritized_replay = prioritized_replay
        self.alpha = alpha  # Controls how much prioritization is used (0 = uniform, 1 = full prioritization)
        self.beta = beta    # Controls importance sampling (starts low, increases to 1)
        self.beta_increment = beta_increment
        self.epsilon_prio = 1e-6  # Small constant to ensure no experience has 0 priority
        
        # Initialize memory
        if self.prioritized_replay:
            # For prioritized replay: (state, action, reward, next_state, done, priority)
            self.memory = []
            self.priorities = np.zeros(memory_size, dtype=np.float32)
            self.memory_pos = 0  # Position to insert next experience
            self.memory_full = False
        else:
            # Regular replay buffer
            self.memory = deque(maxlen=memory_size)
            
        # Set up device (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create networks
        self.policy_net = DQN(state_size, action_size, dueling=dueling_dqn).to(self.device)
        self.target_net = DQN(state_size, action_size, dueling=dueling_dqn).to(self.device)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Set up optimizer (Adam with adjusted learning rate)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='none')  # Use 'none' to apply importance sampling weights
        
        # Track training stats
        self.loss_history = []
        self.steps_done = 0
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        if self.prioritized_replay:
            # For first experiences, use max priority
            max_priority = np.max(self.priorities) if self.memory else 1.0
            
            # Store experience
            experience = (state, action, reward, next_state, done)
            
            # Add to memory with maximum priority for new experiences
            if len(self.memory) < self.memory_size:
                self.memory.append(experience)
                self.priorities[len(self.memory)-1] = max_priority
            else:
                # Memory full, start overwriting
                self.memory[self.memory_pos] = experience
                self.priorities[self.memory_pos] = max_priority
                self.memory_pos = (self.memory_pos + 1) % self.memory_size
                self.memory_full = True
        else:
            # Regular experience replay
            self.memory.append((state, action, reward, next_state, done))
    
    def get_memory_length(self):
        """Get current memory size"""
        if self.prioritized_replay:
            return len(self.memory)
        else:
            return len(self.memory)
    
    def act(self, state):
        """Select an action using epsilon-greedy policy"""
        # Decay epsilon
        self.steps_done += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Explore: choose random action with probability epsilon
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploit: choose best action from policy network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.policy_net.eval()
            action_values = self.policy_net(state_tensor)
            self.policy_net.train()
            return torch.argmax(action_values).item()
    
    def compute_td_error(self, state, action, reward, next_state, done):
        """Compute the TD error for a single transition (for prioritized replay)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Current Q-value estimate
            current_q = self.policy_net(state_tensor)[0][action]
            
            if done:
                target_q = reward
            else:
                if self.double_dqn:
                    # DDQN: use policy net to select action, target net to evaluate it
                    next_action = self.policy_net(next_state_tensor).argmax(1).item()
                    next_q = self.target_net(next_state_tensor)[0][next_action]
                else:
                    # Regular DQN: use target net for both selection and evaluation
                    next_q = self.target_net(next_state_tensor).max(1)[0].item()
                    
                target_q = reward + self.gamma * next_q
                
            # Return absolute TD error
            return abs(target_q - current_q).item()
        
    def sample_batch(self, batch_size):
        """Sample a batch of experiences, using priorities if enabled"""
        if self.prioritized_replay:
            # Get memory size
            memory_size = len(self.memory)
            if memory_size == self.memory_size:
                memory_size = self.memory_size
            
            # Calculate sampling probabilities
            priorities = self.priorities[:memory_size]
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            # Sample indices based on priorities
            indices = np.random.choice(memory_size, batch_size, p=probabilities, replace=False)
            
            # Get experiences from sampled indices
            batch = [self.memory[idx] for idx in indices]
            
            # Calculate importance sampling weights
            weights = (memory_size * probabilities[indices]) ** -self.beta
            weights /= weights.max()  # Normalize weights
            
            # Increase beta toward 1 over time
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            return batch, indices, weights
        else:
            # Regular uniform sampling
            batch = random.sample(self.memory, batch_size)
            return batch, None, None
        
    def replay(self, batch_size):
        """Train the agent by replaying past experiences"""
        # Only train if we have enough experiences
        memory_length = self.get_memory_length()
        if memory_length < batch_size:
            return 0  # Return 0 loss if not enough experiences
        
        # Sample a batch of experiences
        if self.prioritized_replay:
            minibatch, indices, weights = self.sample_batch(batch_size)
            weights_tensor = torch.FloatTensor(weights).to(self.device)
        else:
            minibatch = random.sample(self.memory, batch_size)
            indices, weights_tensor = None, None
            
        # Extract batch components
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q values from policy network for the taken actions
        self.policy_net.train()
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        
        # Compute target Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: Use policy net to select actions and target net to evaluate them
                next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
                next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze()
            else:
                # Regular DQN: Use target net for both selection and evaluation
                next_q_values = self.target_net(next_state_batch).max(1)[0]
                
            # Target Q = reward if done, otherwise reward + gamma * max Q(s',a)
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss (MSE or weighted MSE for prioritized replay)
        loss = self.criterion(q_values, target_q_values)
        
        # Apply importance sampling weights if using prioritized replay
        if self.prioritized_replay:
            weighted_loss = (loss * weights_tensor).mean()
        else:
            weighted_loss = loss.mean()
        
        # Update network weights
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update priorities for sampled experiences if using prioritized replay
        if self.prioritized_replay and indices is not None:
            for i, idx in enumerate(indices):
                # Update priorities with TD errors
                td_error = abs(q_values[i].item() - target_q_values[i].item()) + self.epsilon_prio
                self.priorities[idx] = td_error
        
        # Periodically update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Track loss for monitoring
        loss_value = weighted_loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value

def find_latest_checkpoint_level():
    """Find the highest difficulty level with an existing checkpoint"""
    max_level = 0
    for file in os.listdir("modelCheckpoints"):
        if file.startswith('cube_solver_model_scramble_') and file.endswith('.pt'):
            try:
                # Extract scramble level from filename
                level = int(file.split('_')[-1].split('.')[0])
                max_level = max(max_level, level)
            except ValueError:
                continue
    return max_level

def ensure_scrambles_exist(scramble_moves, use_pregenerated=False):
    """
    Make sure scramble files exist for the given difficulty level.
    Returns True if scrambles exist or were successfully generated, False otherwise.
    """
    if not use_pregenerated:
        return True  # No need to check if not using pregenerated scrambles
        
    # Check if scramble file exists
    filepath = os.path.join("scrambles", f"{scramble_moves}movescramble.txt")
    if os.path.exists(filepath):
        # Check if file has content
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            if first_line:  # File has content
                return True
    
    # File doesn't exist or is empty, generate scrambles
    print(f"Generating scrambles for difficulty level {scramble_moves}...")
    try:
        subprocess.run(["python", "gen.py", "--level", str(scramble_moves)], check=True)
        if os.path.exists(filepath):
            return True
        else:
            print(f"Failed to generate scramble file for level {scramble_moves}.")
            return False
    except Exception as e:
        print(f"Error generating scrambles for level {scramble_moves}: {e}")
        return False

def log_training_results(scramble_moves, episode, solved_episodes, total_episodes, success_rate, recent_success_rate, time_taken, agent_config=None, level_stats=None, recent_scrambles=None, avg_reward=None, avg_loss=None, epsilon=None, memory_size=None, best_success_rate=None, episodes_since_improvement=None, stopping_reason=None):
    """Log training results to a file in the output directory"""
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Create timestamp for the log file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = f"output/training_log_{scramble_moves}moves_{timestamp}.txt"
    
    # Format the log content
    log_content = f"""
=== Training Results ===
Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}
Difficulty Level: {scramble_moves} moves
Episodes: {episode}
Solved Episodes: {solved_episodes}/{total_episodes}
Success Rate: {success_rate:.2f}%
Recent Success Rate: {recent_success_rate:.2f}%
Training Time: {time_taken:.2f} seconds
"""
    
    # Add stopping reason if available
    if stopping_reason:
        log_content += f"Stopping Reason: {stopping_reason}\n"
    
    # Add performance metrics if available
    if avg_reward is not None:
        log_content += f"Average Reward: {avg_reward:.2f}\n"
    if avg_loss is not None:
        log_content += f"Average Loss: {avg_loss:.4f}\n"
    if epsilon is not None:
        log_content += f"Exploration Rate (Epsilon): {epsilon:.4f}\n"
    if memory_size is not None:
        log_content += f"Memory Size: {memory_size}\n"
    if best_success_rate is not None:
        log_content += f"Best Success Rate: {best_success_rate:.2f}%\n"
    if episodes_since_improvement is not None:
        log_content += f"Episodes Since Last Improvement: {episodes_since_improvement}\n"
    
    # Add level statistics if available
    if level_stats:
        log_content += "\nLevel Statistics:\n"
        for stat in level_stats:
            log_content += f"{stat}\n"
    
    # Add recent scrambles if available
    if recent_scrambles:
        log_content += f"\nLast {len(recent_scrambles)} Scrambles:\n"
        for i, (scr, out, mvs, rew) in enumerate(recent_scrambles):
            log_content += f"- {scr} ({out} in {mvs} moves, reward: {rew:.1f})\n"
    
    # Add agent configuration if provided
    if agent_config:
        log_content += "\nAgent Configuration:\n"
        for key, value in agent_config.items():
            log_content += f"  - {key}: {value}\n"
    
    # Write to log file
    with open(log_file, "w") as f:
        f.write(log_content)
    
    print(f"\nTraining results logged to: {log_file}")
    return log_file

def train_specific_level(scramble_moves, min_episodes=5000, max_episodes=10000, 
                         target_success_rate=30, min_success_rate=None, batch_size=64, prev_checkpoint=None, 
                         use_pregenerated=False, recent_window=1000, agent_config=None):
    """
    Train on a specific difficulty level
    
    Args:
        scramble_moves: Number of scramble moves for this difficulty level
        min_episodes: Minimum number of episodes to train for
        max_episodes: Maximum number of episodes to train for
        target_success_rate: Target success rate to achieve before stopping training
        min_success_rate: Minimum success rate to achieve regardless of episode count
        batch_size: Batch size for training
        prev_checkpoint: Path to previous checkpoint to load
        use_pregenerated: Whether to use pregenerated scrambles
        recent_window: Number of recent episodes to consider for success rate calculation
        agent_config: Dictionary of DQNAgent configuration parameters
    """
    print(f"\n=== Starting training with {scramble_moves} scramble moves ===")
    
    # If min_success_rate is not specified, use target_success_rate
    if min_success_rate is None:
        min_success_rate = target_success_rate
    
    # For very high target rates (>90%), disable early stopping to ensure we reach the target
    disable_early_stopping = target_success_rate >= 90

    # If target rate is extremely high (99%+), use a small value for better precision
    if target_success_rate >= 99:
        min_success_rate = target_success_rate - 0.5  # Within 0.5% is acceptable
        print(f"High target success rate detected ({target_success_rate}%). Training will continue until reaching at least {min_success_rate}%")
        if disable_early_stopping:
            print("Early stopping disabled for high target success rate.")
    
    # Ensure scrambles exist if using pregenerated scrambles
    if use_pregenerated:
        if not ensure_scrambles_exist(scramble_moves, use_pregenerated):
            print(f"Warning: Could not generate scrambles for level {scramble_moves}. Disabling pregenerated scrambles.")
            use_pregenerated = False
    
    # Create the primary environment with current difficulty
    main_env = CubeEnvironment(scramble_moves=scramble_moves, use_pregenerated=use_pregenerated)
    
    # Create a set of environments with all difficulties up to current
    all_envs = {}
    for i in range(1, scramble_moves + 1):
        # Ensure scrambles exist for each difficulty level we'll use
        if use_pregenerated and not ensure_scrambles_exist(i, use_pregenerated):
            print(f"Warning: Could not generate scrambles for level {i}. Using random scrambles for this level.")
            all_envs[i] = CubeEnvironment(scramble_moves=i, use_pregenerated=False)
        else:
            all_envs[i] = CubeEnvironment(scramble_moves=i, use_pregenerated=use_pregenerated)
    
    state_size = 6 * 9 * 6  # 6 faces, 9 stickers per face, 6 possible colors
    action_size = len(MOVES)
    
    # Set default agent configuration if not provided
    default_config = {
        'learning_rate': 0.001,
        'memory_size': 100000,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.995,
        'target_update_freq': 1000,
        'double_dqn': True,
        'dueling_dqn': True,
        'prioritized_replay': True,
        'alpha': 0.6,
        'beta': 0.4,
        'beta_increment': 0.001,
        'plateau_required': 5  # Default value for plateau detection
    }
    
    # Override defaults with provided configuration
    if agent_config:
        default_config.update(agent_config)
    
    # Extract plateau detection parameter
    plateau_required = default_config.get('plateau_required', 5)
    print(f"Plateau detection will require {plateau_required} stable measurements")
    
    # Initialize agent with configuration
    agent = DQNAgent(
        state_size=state_size, 
        action_size=action_size,
        learning_rate=default_config['learning_rate'],
        memory_size=default_config['memory_size'],
        gamma=default_config['gamma'],
        epsilon_start=default_config['epsilon_start'],
        epsilon_min=default_config['epsilon_min'],
        epsilon_decay=default_config['epsilon_decay'],
        target_update_freq=default_config['target_update_freq'],
        double_dqn=default_config['double_dqn'],
        dueling_dqn=default_config['dueling_dqn'],
        prioritized_replay=default_config['prioritized_replay'],
        alpha=default_config['alpha'],
        beta=default_config['beta'],
        beta_increment=default_config['beta_increment']
    )
    
    # Print agent configuration
    print(f"Agent Configuration:")
    for key, value in default_config.items():
        print(f"  - {key}: {value}")
    
    # Load previous checkpoint if available
    if prev_checkpoint:
        try:
            agent.policy_net.load_state_dict(torch.load(prev_checkpoint))
            # Also load to target network
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"Loaded checkpoint from {prev_checkpoint}")
            
            # Use a lower epsilon when using a checkpoint to better utilize learned knowledge
            # For the same level, use very little exploration
            if os.path.basename(prev_checkpoint) == f'cube_solver_model_scramble_{scramble_moves}.pt':
                agent.epsilon = max(0.05, agent.epsilon_min)  # Minimal exploration when continuing same level
                print(f"Continuing training on same level {scramble_moves} with minimal exploration (epsilon: {agent.epsilon:.4f})")
            else:
                # For a new difficulty level, use moderate exploration 
                agent.epsilon = 0.2  # 20% exploration when moving to a new level
                print(f"Training level {scramble_moves} from previous level checkpoint with moderate exploration (epsilon: {agent.epsilon:.4f})")
        except Exception as e:
            print(f"Failed to load checkpoint from {prev_checkpoint}, starting fresh: {e}")
    
    solved_episodes = 0
    solved_by_difficulty = {i: 0 for i in range(1, scramble_moves + 1)}
    episodes_by_difficulty = {i: 0 for i in range(1, scramble_moves + 1)}
    start_time = time.time()
    current_success_rate = 0
    recent_success_rate = 0
    episode = 0
    
    # Keep track of recent scrambles and outcomes
    recent_scrambles = []  # Will store tuples of (scramble, outcome, moves)
    
    # Keep track of recent episodes for success rate calculation
    recent_episodes = deque(maxlen=recent_window)
    recent_solved = deque(maxlen=recent_window)
    
    # Track training metrics
    training_losses = []
    avg_rewards = deque(maxlen=100)
    
    # Define the final checkpoint name for this difficulty level
    final_checkpoint = os.path.join("modelCheckpoints", f'cube_solver_model_scramble_{scramble_moves}.pt')
    best_checkpoint = os.path.join("modelCheckpoints", f'cube_solver_model_scramble_{scramble_moves}_best.pt')
    
    # Variables for tracking best model
    best_success_rate = 0
    episodes_since_improvement = 0
    patience = 500  # Number of episodes to wait for improvement before early stopping
    
    # Smart plateau detection parameters
    success_rate_history = []  # Track success rates over time
    plateau_window = 10  # Check last 10 chunks of 500 episodes each
    plateau_chunk_size = 500  # Each chunk represents 500 episodes
    plateau_threshold = 0.3  # Required improvement percentage points between chunks
    stable_rate_count = 0  # Count of stable measurements
    stable_rate_required = plateau_required  # Use the configurable parameter
    
    # Training loop - continue until we reach target success rate or max episodes
    # Also ensure we continue training if we haven't reached min_success_rate
    while (episode < max_episodes or (recent_success_rate < min_success_rate)):
        episode += 1
        
        # Early stopping if no improvement for a long time and not disabled
        if not disable_early_stopping and episodes_since_improvement > patience and episode > min_episodes:
            print(f"No improvement for {patience} episodes. Early stopping at episode {episode}.")
            print(f"Best success rate achieved: {best_success_rate:.2f}% (target was {target_success_rate}%)")
            break
            
        # Distribution of difficulties:
        # 30% current difficulty, 50% previous level, 20% distributed among easier difficulties
        if scramble_moves == 1:
            # For level 1, always use current difficulty
            selected_diff = scramble_moves
        elif random.random() < 0.3:
            # 30% chance to use current difficulty
            selected_diff = scramble_moves
        elif random.random() < 0.8 and scramble_moves > 1:
            # 50% chance to use previous level (when available)
            selected_diff = scramble_moves - 1
        else:
            # 20% chance to select from easier difficulties with preference toward harder ones
            weights = [i/sum(range(1, scramble_moves-1)) for i in range(1, scramble_moves-1)]
            selected_diff = random.choices(range(1, scramble_moves-1), weights=weights)[0]
        
        # Use the environment with selected difficulty
        env_to_use = all_envs[selected_diff]
        episodes_by_difficulty[selected_diff] += 1
        
        state = env_to_use.reset()
        done = False
        moves_taken = 0
        episode_reward = 0
        
        # Get the scramble that was applied
        scramble = " ".join(env_to_use.scramble_sequence) if env_to_use.scramble_sequence else "Random scramble"
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env_to_use.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            moves_taken += 1
            episode_reward += reward
            
            if done:
                solved = env_to_use.cube.is_solved()
                outcome = "Solved" if solved else "Failed"
                # Add to recent scrambles
                recent_scrambles.append((scramble, outcome, moves_taken, episode_reward))
                # Keep only the 5 most recent scrambles
                if len(recent_scrambles) > 5:
                    recent_scrambles.pop(0)
                
                if solved:
                    solved_by_difficulty[selected_diff] += 1
                    if selected_diff == scramble_moves:  # Count solutions for current difficulty
                        solved_episodes += 1
                        
                # Track recent episodes (only for the current difficulty level)
                if selected_diff == scramble_moves:
                    recent_episodes.append(1)
                    recent_solved.append(1 if solved else 0)
                    avg_rewards.append(episode_reward)
                break
        
        # Train the agent with experiences from memory
        loss = agent.replay(batch_size)
        training_losses.append(loss)
        
        # Print progress and check success rate every 100 episodes
        if episode % 100 == 0 or episode == 1:
            # Calculate success rate for current difficulty
            main_episodes = episodes_by_difficulty[scramble_moves]
            main_solved = solved_by_difficulty[scramble_moves]
            
            if main_episodes > 0:
                current_success_rate = (main_solved / main_episodes) * 100
            
            # Calculate recent success rate
            if len(recent_episodes) > 0:
                recent_success_rate = (sum(recent_solved) / len(recent_episodes)) * 100
            else:
                recent_success_rate = 0
            
            # Calculate average reward
            avg_reward = sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0
            
            all_solved = sum(solved_by_difficulty.values())
            all_episodes = sum(episodes_by_difficulty.values())
            overall_rate = (all_solved / all_episodes) * 100
            
            # Update best model if improved
            if recent_success_rate > best_success_rate and len(recent_episodes) >= 200:
                best_success_rate = recent_success_rate
                torch.save(agent.policy_net.state_dict(), best_checkpoint)
                print(f"\nNew best model saved with success rate: {best_success_rate:.2f}%")
                episodes_since_improvement = 0
            else:
                episodes_since_improvement += 100
            
            # Store success rate history for plateau detection (every 500 episodes)
            if episode % 500 == 0 and len(recent_episodes) >= 300:
                success_rate_history.append(recent_success_rate)
                
                # Only start checking for plateau after we have enough data points
                if len(success_rate_history) >= plateau_window:
                    # Keep only the most recent measurements
                    if len(success_rate_history) > plateau_window:
                        success_rate_history = success_rate_history[-plateau_window:]
                    
                    # Check if success rate has plateaued
                    improving = False
                    
                    # Calculate average improvement over the last several chunks
                    improvements = [success_rate_history[i] - success_rate_history[i-1] 
                                   for i in range(1, len(success_rate_history))]
                    avg_improvement = sum(improvements) / len(improvements)
                    
                    # Check the trend in the last few measurements
                    recent_improvements = improvements[-3:] if len(improvements) >= 3 else improvements
                    recent_avg_improvement = sum(recent_improvements) / len(recent_improvements)
                    
                    # Detect if we're improving at a meaningful rate
                    if recent_avg_improvement >= plateau_threshold or avg_improvement >= plateau_threshold * 0.7:
                        improving = True
                        stable_rate_count = 0  # Reset stability counter if still improving
                    
                    # If we've reached a good success rate and improvement has plateaued
                    if not improving and recent_success_rate >= 50:  # Only consider plateau if success rate is decent
                        stable_rate_count += 1
                        
                        # Log plateau detection progress
                        remaining = stable_rate_required - stable_rate_count
                        if remaining > 0:
                            print(f"\nPlateau detection: Success rate has stabilized around {recent_success_rate:.2f}%")
                            print(f"Recent improvement rate: {recent_avg_improvement:.2f}% per {plateau_chunk_size} episodes")
                            print(f"Will confirm plateau in {remaining} more measurement{'s' if remaining > 1 else ''}")
                        
                        # If we've confirmed a plateau and no significant improvement for a while
                        if stable_rate_count >= stable_rate_required:
                            # If we're already at a good success rate, consider this the maximum achievable
                            if recent_success_rate >= 75:  # 75% is generally a good success rate
                                print(f"\n=== PLATEAU DETECTED ===")
                                print(f"Success rate has stabilized at {recent_success_rate:.2f}% after {episode} episodes")
                                print(f"This appears to be the maximum achievable rate for level {scramble_moves}")
                                print(f"Target rate of {target_success_rate}% may not be achievable for this difficulty")
                                print(f"Stopping training and saving best model with success rate: {best_success_rate:.2f}%")
                                
                                # Update the log file to indicate why we stopped
                                plateau_reason = f"Training stopped due to plateau detection. Maximum achievable rate appears to be ~{recent_success_rate:.2f}%"
                                
                                break
                    else:
                        stable_rate_count = 0  # Reset if we're still improving or success rate is too low
            
            # Clear screen for cleaner display
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Display training progress
            print(f"=== Training Progress === - Level: {scramble_moves}")
            print(f"  - Episode: {episode}/{max_episodes if episode <= max_episodes else 'unlimited until '+str(min_success_rate)+'% success'}")
            print(f"  - Min Episodes: {min_episodes} - Max Episodes: {max_episodes}")
            print(f"  - Target Success Rate: {target_success_rate}% - Min Success Rate: {min_success_rate}%")
            print(f"  - Batch Size: {batch_size} - Using Pregenerated Scrambles: {use_pregenerated}")
            print(f"  - Previous Checkpoint: {prev_checkpoint if prev_checkpoint else 'None'}")
            print("=" * 50)
            
            # Performance metrics
            print(f"Performance Summary:")
            print(f"✓ Overall Success Rate: {current_success_rate:.2f}%")
            print(f"✓ Recent Success Rate (Last {len(recent_episodes)} Episodes): {recent_success_rate:.2f}%")
            print(f"✓ Best Success Rate: {best_success_rate:.2f}%") 
            print(f"✓ Average Reward: {avg_reward:.2f}")
            print(f"✓ Average Loss: {sum(training_losses[-100:]) / len(training_losses[-100:]) if training_losses else 0:.4f}")
            print(f"✓ Exploration Rate (Epsilon): {agent.epsilon:.4f}")
            print(f"✓ Memory Size: {agent.get_memory_length()}")
            print(f"✓ Episodes Since Last Improvement: {episodes_since_improvement}")
            print()
            
            # Level statistics
            print(f"Level Statistics:")
            diff_stats = [f"D{d}: {solved_by_difficulty[d]}/{episodes_by_difficulty[d]} "
                         f"({(solved_by_difficulty[d]/episodes_by_difficulty[d])*100:.1f}%)" 
                         for d in sorted(solved_by_difficulty.keys()) 
                         if episodes_by_difficulty[d] > 0]
            for stat in diff_stats:
                print(stat)
            print()
            
            # Recent scrambles
            print(f"Last {len(recent_scrambles)} Scrambles:")
            for i, (scr, out, mvs, rew) in enumerate(recent_scrambles):
                print(f"- {scr} ({out} in {mvs} moves, reward: {rew:.1f})")
                
            # Save intermediate checkpoint, replacing the previous one
            if episode % 1000 == 0:
                torch.save(agent.policy_net.state_dict(), final_checkpoint)
                print(f"\nSaved checkpoint: {final_checkpoint}")
        
        # Check if we've met the success criteria and minimum episodes
        # Use recent success rate instead of overall success rate
        # And we must have enough episodes to have a meaningful measurement
        if episode >= min_episodes and len(recent_episodes) >= min(recent_window, 500):
            if recent_success_rate >= target_success_rate:
                print(f"\nReached target success rate of {target_success_rate}% (actual {recent_success_rate:.2f}%) after {episode} episodes!")
                break
            elif recent_success_rate >= min_success_rate and best_success_rate >= target_success_rate:
                # If we've reached the minimum required rate and previously hit the target rate
                print(f"\nReached minimum success rate of {min_success_rate}% with best at {best_success_rate:.2f}% (target: {target_success_rate}%) after {episode} episodes!")
                break
    
    # Save final model for this scramble difficulty
    torch.save(agent.policy_net.state_dict(), final_checkpoint)
    total_time = time.time() - start_time
    
    # Determine stopping reason for logging
    if 'plateau_reason' in locals():
        stopping_reason = plateau_reason
    elif recent_success_rate >= target_success_rate:
        stopping_reason = f"Reached target success rate of {target_success_rate}%"
    elif episodes_since_improvement > patience and not disable_early_stopping:
        stopping_reason = f"Early stopping due to no improvement for {patience} episodes"
    elif episode >= max_episodes:
        stopping_reason = f"Reached maximum number of episodes: {max_episodes}"
    else:
        stopping_reason = "Training completed normally"
    
    print(f"\nTraining completed. Scramble moves: {scramble_moves}, "
          f"Episodes: {episode}, Solved: {solved_episodes}/{episodes_by_difficulty[scramble_moves]}, "
          f"Success Rate: {current_success_rate:.2f}%, Recent Success Rate: {recent_success_rate:.2f}%, "
          f"Time: {total_time:.2f} seconds.")
    print(f"Stopping reason: {stopping_reason}")
    
    # Calculate level statistics for logging
    level_stats = [f"D{d}: {solved_by_difficulty[d]}/{episodes_by_difficulty[d]} "
                   f"({(solved_by_difficulty[d]/episodes_by_difficulty[d])*100:.1f}%)" 
                   for d in sorted(solved_by_difficulty.keys()) 
                   if episodes_by_difficulty[d] > 0]
    
    # Calculate average loss for logging
    avg_loss = sum(training_losses[-100:]) / len(training_losses[-100:]) if training_losses else 0
    
    # Calculate average reward for logging
    avg_reward = sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0
    
    # Log training results to file
    log_file = log_training_results(
        scramble_moves=scramble_moves,
        episode=episode,
        solved_episodes=solved_episodes,
        total_episodes=episodes_by_difficulty[scramble_moves],
        success_rate=current_success_rate,
        recent_success_rate=recent_success_rate,
        time_taken=total_time,
        agent_config=default_config,
        level_stats=level_stats,
        recent_scrambles=recent_scrambles,
        avg_reward=avg_reward,
        avg_loss=avg_loss,
        epsilon=agent.epsilon,
        memory_size=agent.get_memory_length(),
        best_success_rate=best_success_rate,
        episodes_since_improvement=episodes_since_improvement,
        stopping_reason=stopping_reason
    )
    
    # Use the best model if we have one
    if os.path.exists(best_checkpoint) and best_success_rate > current_success_rate:
        print(f"Loading best model with success rate {best_success_rate:.2f}% for testing")
        agent.policy_net.load_state_dict(torch.load(best_checkpoint))
        # Copy best model to final checkpoint
        torch.save(agent.policy_net.state_dict(), final_checkpoint)
    
    # Test on current difficulty
    print(f"\n=== Testing agent with {scramble_moves} scramble moves ===")
    test_agent(num_tests=50, scramble_moves=scramble_moves, checkpoint_path=final_checkpoint, use_pregenerated=use_pregenerated)
    
    return final_checkpoint

def train_mixed_levels(level1, level2, mix_ratio=0.5, min_episodes=5000, max_episodes=10000, 
                       target_success_rate=30, min_success_rate=None, batch_size=64, prev_checkpoint=None, 
                       use_pregenerated=False, recent_window=1000, agent_config=None):
    """
    Train on a mix of two difficulty levels.
    
    Args:
        level1: First difficulty level (lower)
        level2: Second difficulty level (higher)
        mix_ratio: Ratio of level2 vs level1 (0.5 means 50% level2, 50% level1)
        min_episodes: Minimum number of episodes to train for
        max_episodes: Maximum number of episodes to train for
        target_success_rate: Target success rate to achieve before stopping training
        min_success_rate: Minimum success rate to achieve regardless of episode count
        batch_size: Batch size for training
        prev_checkpoint: Path to previous checkpoint to load
        use_pregenerated: Whether to use pregenerated scrambles
        recent_window: Number of recent episodes to consider for success rate calculation
        agent_config: Dictionary of DQNAgent configuration parameters
    """
    print(f"\n=== Starting mixed training with {level1} and {level2} scramble moves (ratio: {mix_ratio:.2f}) ===")
    
    # If min_success_rate is not specified, use target_success_rate
    if min_success_rate is None:
        min_success_rate = target_success_rate
    
    # For very high target rates (>90%), disable early stopping to ensure we reach the target
    disable_early_stopping = target_success_rate >= 90

    # Ensure scrambles exist if using pregenerated scrambles
    if use_pregenerated:
        for level in [level1, level2]:
            if not ensure_scrambles_exist(level, use_pregenerated):
                print(f"Warning: Could not generate scrambles for level {level}. Using random scrambles.")
                use_pregenerated = False
    
    # Create environments for both difficulty levels
    env1 = CubeEnvironment(scramble_moves=level1, use_pregenerated=use_pregenerated)
    env2 = CubeEnvironment(scramble_moves=level2, use_pregenerated=use_pregenerated)
    
    # Create a set of environments with all difficulties up to the higher level
    all_envs = {}
    for i in range(1, max(level1, level2) + 1):
        # Ensure scrambles exist for each difficulty level we'll use
        if use_pregenerated and not ensure_scrambles_exist(i, use_pregenerated):
            print(f"Warning: Could not generate scrambles for level {i}. Using random scrambles for this level.")
            all_envs[i] = CubeEnvironment(scramble_moves=i, use_pregenerated=False)
        else:
            all_envs[i] = CubeEnvironment(scramble_moves=i, use_pregenerated=use_pregenerated)
    
    state_size = 6 * 9 * 6  # 6 faces, 9 stickers per face, 6 possible colors
    action_size = len(MOVES)
    
    # Set default agent configuration if not provided
    default_config = {
        'learning_rate': 0.001,
        'memory_size': 100000,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.995,
        'target_update_freq': 1000,
        'double_dqn': True,
        'dueling_dqn': True,
        'prioritized_replay': True,
        'alpha': 0.6,
        'beta': 0.4,
        'beta_increment': 0.001,
        'plateau_required': 5  # Default value for plateau detection
    }
    
    # Override defaults with provided configuration
    if agent_config:
        default_config.update(agent_config)
    
    # Extract plateau detection parameter
    plateau_required = default_config.get('plateau_required', 5)
    print(f"Plateau detection will require {plateau_required} stable measurements")
    
    # Initialize agent with configuration
    agent = DQNAgent(
        state_size=state_size, 
        action_size=action_size,
        learning_rate=default_config['learning_rate'],
        memory_size=default_config['memory_size'],
        gamma=default_config['gamma'],
        epsilon_start=default_config['epsilon_start'],
        epsilon_min=default_config['epsilon_min'],
        epsilon_decay=default_config['epsilon_decay'],
        target_update_freq=default_config['target_update_freq'],
        double_dqn=default_config['double_dqn'],
        dueling_dqn=default_config['dueling_dqn'],
        prioritized_replay=default_config['prioritized_replay'],
        alpha=default_config['alpha'],
        beta=default_config['beta'],
        beta_increment=default_config['beta_increment']
    )
    
    # Print agent configuration
    print(f"Agent Configuration:")
    for key, value in default_config.items():
        print(f"  - {key}: {value}")
    
    # Load previous checkpoint if available
    if prev_checkpoint:
        try:
            agent.policy_net.load_state_dict(torch.load(prev_checkpoint))
            # Also load to target network
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"Loaded checkpoint from {prev_checkpoint}")
            
            # Use a lower epsilon when using a checkpoint to better utilize learned knowledge
            agent.epsilon = 0.2  # 20% exploration when starting from a checkpoint
            print(f"Training with mixed levels using moderate exploration (epsilon: {agent.epsilon:.4f})")
        except Exception as e:
            print(f"Failed to load checkpoint from {prev_checkpoint}, starting fresh: {e}")
    
    # Track episodes and success rates for both levels
    solved_episodes = {level1: 0, level2: 0}
    episodes_by_difficulty = {level1: 0, level2: 0}
    
    # Track all difficulties (including curriculum learning)
    all_solved_by_difficulty = {i: 0 for i in range(1, max(level1, level2) + 1)}
    all_episodes_by_difficulty = {i: 0 for i in range(1, max(level1, level2) + 1)}
    
    start_time = time.time()
    episode = 0
    
    # Keep track of recent scrambles and outcomes
    recent_scrambles = []  # Will store tuples of (scramble, outcome, moves)
    
    # Keep track of recent episodes for success rate calculation
    # Separate tracking for each level
    recent_episodes = {
        level1: deque(maxlen=recent_window),
        level2: deque(maxlen=recent_window)
    }
    recent_solved = {
        level1: deque(maxlen=recent_window),
        level2: deque(maxlen=recent_window)
    }
    
    # Track training metrics
    training_losses = []
    avg_rewards = deque(maxlen=100)
    
    # Define the checkpoint names
    mixed_name = f"{level1}_{level2}_mix{int(mix_ratio*100)}"
    final_checkpoint = os.path.join("modelCheckpoints", f'cube_solver_model_mixed_{mixed_name}.pt')
    best_checkpoint = os.path.join("modelCheckpoints", f'cube_solver_model_mixed_{mixed_name}_best.pt')
    
    # Variables for tracking best model
    best_success_rate = {level1: 0, level2: 0}
    best_combined_rate = 0
    episodes_since_improvement = 0
    patience = 500  # Number of episodes to wait for improvement before early stopping
    
    # Smart plateau detection parameters
    success_rate_history = []  # Track success rates over time
    plateau_window = 10  # Check last 10 chunks of 500 episodes each
    plateau_chunk_size = 500  # Each chunk represents 500 episodes
    plateau_threshold = 0.3  # Required improvement percentage points between chunks
    stable_rate_count = 0  # Count of stable measurements
    stable_rate_required = plateau_required  # Use the configurable parameter
    
    # Training loop
    while (episode < max_episodes or (recent_success_rate_level2 < min_success_rate)):
        episode += 1
        
        # Early stopping if no improvement for a long time and not disabled
        if not disable_early_stopping and episodes_since_improvement > patience and episode > min_episodes:
            print(f"No improvement for {patience} episodes. Early stopping at episode {episode}.")
            print(f"Best combined success rate achieved: {best_combined_rate:.2f}%")
            break
            
        # Distribute between the two main levels and curriculum learning
        rand = random.random()
        
        if rand < mix_ratio:
            # Use the higher difficulty level
            selected_diff = level2
            
            # 30% chance of using an easier difficulty for curriculum learning when training on level2
            if rand < mix_ratio * 0.3:
                # Select from easier difficulties with preference toward harder ones
                weights = [i/sum(range(1, level2)) for i in range(1, level2)]
                selected_diff = random.choices(range(1, level2), weights=weights)[0]
        else:
            # Use the lower difficulty level
            selected_diff = level1
            
            # 20% chance of using an even easier difficulty when training on level1
            if rand < (1 - mix_ratio) * 0.2 and level1 > 1:
                # Select from easier difficulties with preference toward harder ones
                weights = [i/sum(range(1, level1)) for i in range(1, level1)]
                selected_diff = random.choices(range(1, level1), weights=weights)[0]
        
        # Use the environment with selected difficulty
        env_to_use = all_envs[selected_diff]
        all_episodes_by_difficulty[selected_diff] += 1
        
        # Track episodes for our main levels
        if selected_diff == level1:
            episodes_by_difficulty[level1] += 1
        elif selected_diff == level2:
            episodes_by_difficulty[level2] += 1
        
        state = env_to_use.reset()
        done = False
        moves_taken = 0
        episode_reward = 0
        
        # Get the scramble that was applied
        scramble = " ".join(env_to_use.scramble_sequence) if env_to_use.scramble_sequence else "Random scramble"
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env_to_use.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            moves_taken += 1
            episode_reward += reward
            
            if done:
                solved = env_to_use.cube.is_solved()
                outcome = "Solved" if solved else "Failed"
                
                # Add to recent scrambles
                recent_scrambles.append((scramble, outcome, moves_taken, episode_reward))
                # Keep only the 5 most recent scrambles
                if len(recent_scrambles) > 5:
                    recent_scrambles.pop(0)
                
                # Track overall solve counts by difficulty
                if solved:
                    all_solved_by_difficulty[selected_diff] += 1
                    
                    # Track for our main levels
                    if selected_diff == level1:
                        solved_episodes[level1] += 1
                    elif selected_diff == level2:
                        solved_episodes[level2] += 1
                        
                # Track recent episodes (only for the main difficulty levels)
                if selected_diff == level1:
                    recent_episodes[level1].append(1)
                    recent_solved[level1].append(1 if solved else 0)
                    avg_rewards.append(episode_reward)
                elif selected_diff == level2:
                    recent_episodes[level2].append(1)
                    recent_solved[level2].append(1 if solved else 0)
                    avg_rewards.append(episode_reward)
                break
        
        # Train the agent with experiences from memory
        loss = agent.replay(batch_size)
        training_losses.append(loss)
        
        # Print progress and check success rate every 100 episodes
        if episode % 100 == 0 or episode == 1:
            # Calculate success rates for both main levels
            success_rate_level1 = 0
            success_rate_level2 = 0
            
            if episodes_by_difficulty[level1] > 0:
                success_rate_level1 = (solved_episodes[level1] / episodes_by_difficulty[level1]) * 100
            
            if episodes_by_difficulty[level2] > 0:
                success_rate_level2 = (solved_episodes[level2] / episodes_by_difficulty[level2]) * 100
            
            # Calculate recent success rates
            recent_success_rate_level1 = 0
            recent_success_rate_level2 = 0
            
            if len(recent_episodes[level1]) > 0:
                recent_success_rate_level1 = (sum(recent_solved[level1]) / len(recent_episodes[level1])) * 100
            
            if len(recent_episodes[level2]) > 0:
                recent_success_rate_level2 = (sum(recent_solved[level2]) / len(recent_episodes[level2])) * 100
            
            # Calculate average reward
            avg_reward = sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0
            
            # Calculate overall success rate (weighted by mix ratio)
            combined_rate = ((1-mix_ratio) * recent_success_rate_level1 + 
                             mix_ratio * recent_success_rate_level2)
            
            # Update best model if improved
            improved = False
            
            # Track improvements for each level
            if recent_success_rate_level1 > best_success_rate[level1] and len(recent_episodes[level1]) >= 100:
                best_success_rate[level1] = recent_success_rate_level1
                improved = True
                
            if recent_success_rate_level2 > best_success_rate[level2] and len(recent_episodes[level2]) >= 100:
                best_success_rate[level2] = recent_success_rate_level2
                improved = True
            
            # Calculate combined rate for model saving
            if combined_rate > best_combined_rate and min(len(recent_episodes[level1]), len(recent_episodes[level2])) >= 100:
                best_combined_rate = combined_rate
                torch.save(agent.policy_net.state_dict(), best_checkpoint)
                print(f"\nNew best model saved with combined success rate: {best_combined_rate:.2f}%")
                episodes_since_improvement = 0
            elif improved:
                episodes_since_improvement = 0
            else:
                episodes_since_improvement += 100
            
            # Store success rate history for plateau detection (every 500 episodes)
            if episode % 500 == 0:
                success_rate_history.append(combined_rate)
                
                # Only start checking for plateau after we have enough data points
                if len(success_rate_history) >= plateau_window:
                    # Keep only the most recent measurements
                    if len(success_rate_history) > plateau_window:
                        success_rate_history = success_rate_history[-plateau_window:]
                    
                    # Check if success rate has plateaued
                    improving = False
                    
                    # Calculate average improvement over the last several chunks
                    improvements = [success_rate_history[i] - success_rate_history[i-1] 
                                   for i in range(1, len(success_rate_history))]
                    avg_improvement = sum(improvements) / len(improvements)
                    
                    # Check the trend in the last few measurements
                    recent_improvements = improvements[-3:] if len(improvements) >= 3 else improvements
                    recent_avg_improvement = sum(recent_improvements) / len(recent_improvements)
                    
                    # Detect if we're improving at a meaningful rate
                    if recent_avg_improvement >= plateau_threshold or avg_improvement >= plateau_threshold * 0.7:
                        improving = True
                        stable_rate_count = 0  # Reset stability counter if still improving
                    
                    # If we've reached a good success rate and improvement has plateaued
                    if not improving and combined_rate >= 50:  # Only consider plateau if success rate is decent
                        stable_rate_count += 1
                        
                        # Log plateau detection progress
                        remaining = stable_rate_required - stable_rate_count
                        if remaining > 0:
                            print(f"\nPlateau detection: Success rate has stabilized around {combined_rate:.2f}%")
                            print(f"Recent improvement rate: {recent_avg_improvement:.2f}% per {plateau_chunk_size} episodes")
                            print(f"Will confirm plateau in {remaining} more measurement{'s' if remaining > 1 else ''}")
                        
                        # If we've confirmed a plateau and no significant improvement for a while
                        if stable_rate_count >= stable_rate_required:
                            # If we're already at a good success rate, consider this the maximum achievable
                            if combined_rate >= 75:  # 75% is generally a good success rate
                                print(f"\n=== PLATEAU DETECTED ===")
                                print(f"Success rate has stabilized at {combined_rate:.2f}% after {episode} episodes")
                                print(f"This appears to be the maximum achievable rate")
                                print(f"Target rate of {target_success_rate}% may not be achievable")
                                print(f"Stopping training and saving best model with success rate: {best_combined_rate:.2f}%")
                                
                                # Update the log file to indicate why we stopped
                                plateau_reason = f"Training stopped due to plateau detection. Maximum achievable rate appears to be ~{combined_rate:.2f}%"
                                
                                break
                    else:
                        stable_rate_count = 0  # Reset if we're still improving or success rate is too low
            
            # Clear screen for cleaner display
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Display training progress
            print(f"=== Mixed Training Progress === - Levels: {level1} and {level2} (Mix ratio: {mix_ratio:.2f})")
            print(f"  - Episode: {episode}/{max_episodes if episode <= max_episodes else 'unlimited until '+str(min_success_rate)+'% success'}")
            print(f"  - Min Episodes: {min_episodes} - Max Episodes: {max_episodes}")
            print(f"  - Target Success Rate: {target_success_rate}% - Min Success Rate: {min_success_rate}%")
            print(f"  - Batch Size: {batch_size} - Using Pregenerated Scrambles: {use_pregenerated}")
            print(f"  - Previous Checkpoint: {prev_checkpoint if prev_checkpoint else 'None'}")
            print("=" * 50)
            
            # Performance metrics
            print(f"Performance Summary:")
            print(f"✓ Level {level1} Success Rate: {success_rate_level1:.2f}% (Recent: {recent_success_rate_level1:.2f}%)")
            print(f"✓ Level {level2} Success Rate: {success_rate_level2:.2f}% (Recent: {recent_success_rate_level2:.2f}%)")
            print(f"✓ Combined Success Rate: {combined_rate:.2f}%")
            print(f"✓ Best Combined Success Rate: {best_combined_rate:.2f}%") 
            print(f"✓ Average Reward: {avg_reward:.2f}")
            print(f"✓ Average Loss: {sum(training_losses[-100:]) / len(training_losses[-100:]) if training_losses else 0:.4f}")
            print(f"✓ Exploration Rate (Epsilon): {agent.epsilon:.4f}")
            print(f"✓ Memory Size: {agent.get_memory_length()}")
            print(f"✓ Episodes Since Last Improvement: {episodes_since_improvement}")
            print()
            
            # Level statistics
            print(f"Level Statistics:")
            # Main levels first
            main_stats = [
                f"L{level1}: {solved_episodes[level1]}/{episodes_by_difficulty[level1]} ({(solved_episodes[level1]/episodes_by_difficulty[level1])*100:.1f}% success)" if episodes_by_difficulty[level1] > 0 else f"L{level1}: No episodes yet",
                f"L{level2}: {solved_episodes[level2]}/{episodes_by_difficulty[level2]} ({(solved_episodes[level2]/episodes_by_difficulty[level2])*100:.1f}% success)" if episodes_by_difficulty[level2] > 0 else f"L{level2}: No episodes yet"
            ]
            
            # Then all difficulties
            all_stats = [f"D{d}: {all_solved_by_difficulty[d]}/{all_episodes_by_difficulty[d]} "
                         f"({(all_solved_by_difficulty[d]/all_episodes_by_difficulty[d])*100:.1f}%)" 
                         for d in sorted(all_solved_by_difficulty.keys()) 
                         if all_episodes_by_difficulty[d] > 0]
            
            for stat in main_stats:
                print(stat)
            print("All difficulties:")
            for stat in all_stats:
                print(stat)
            print()
            
            # Recent scrambles
            print(f"Last {len(recent_scrambles)} Scrambles:")
            for i, (scr, out, mvs, rew) in enumerate(recent_scrambles):
                print(f"- {scr} ({out} in {mvs} moves, reward: {rew:.1f})")
                
            # Save intermediate checkpoint, replacing the previous one
            if episode % 1000 == 0:
                torch.save(agent.policy_net.state_dict(), final_checkpoint)
                print(f"\nSaved checkpoint: {final_checkpoint}")
        
        # Check if we've met the success criteria
        if episode >= min_episodes:
            # Calculate the current combined rate
            current_combined_rate = 0
            
            # Make sure we have enough episodes to have a meaningful measurement
            if len(recent_episodes[level1]) >= 200 and len(recent_episodes[level2]) >= 200:
                current_combined_rate = ((1-mix_ratio) * sum(recent_solved[level1]) / len(recent_episodes[level1]) * 100 + 
                                     mix_ratio * sum(recent_solved[level2]) / len(recent_episodes[level2]) * 100)
                                     
                # We prioritize level2 success rate but also need a decent level1 rate
                current_success_rate_level2 = sum(recent_solved[level2]) / len(recent_episodes[level2]) * 100
                current_success_rate_level1 = sum(recent_solved[level1]) / len(recent_episodes[level1]) * 100
                
                # Check if success criteria are met
                if (current_success_rate_level2 >= target_success_rate and
                    current_success_rate_level1 >= 75):  # Level1 should be quite high
                    print(f"\nReached target success rate on both levels! Level {level1}: {current_success_rate_level1:.2f}%, Level {level2}: {current_success_rate_level2:.2f}%")
                    break
                elif (current_success_rate_level2 >= min_success_rate and 
                      current_success_rate_level1 >= 70 and
                      best_combined_rate >= target_success_rate):
                    # If we've reached minimum rate and previously hit the target rate
                    print(f"\nReached minimum success rates with best at {best_combined_rate:.2f}% (target: {target_success_rate}%)")
                    break
    
    # Save final model
    torch.save(agent.policy_net.state_dict(), final_checkpoint)
    total_time = time.time() - start_time
    
    # Determine stopping reason for logging
    if 'plateau_reason' in locals():
        stopping_reason = plateau_reason
    elif combined_rate >= target_success_rate:
        stopping_reason = f"Reached target success rate of {target_success_rate}%"
    elif episodes_since_improvement > patience and not disable_early_stopping:
        stopping_reason = f"Early stopping due to no improvement for {patience} episodes"
    elif episode >= max_episodes:
        stopping_reason = f"Reached maximum number of episodes: {max_episodes}"
    else:
        stopping_reason = "Training completed normally"
    
    print(f"\nMixed training completed. Levels: {level1} and {level2}, "
          f"Episodes: {episode}, Level {level1} Success: {success_rate_level1:.2f}%, "
          f"Level {level2} Success: {success_rate_level2:.2f}%, "
          f"Time: {total_time:.2f} seconds.")
    print(f"Stopping reason: {stopping_reason}")
    
    # Calculate level statistics for logging
    level_stats = []
    # Main levels first  
    if episodes_by_difficulty[level1] > 0:
        level_stats.append(f"Level {level1}: {solved_episodes[level1]}/{episodes_by_difficulty[level1]} "
                    f"({(solved_episodes[level1]/episodes_by_difficulty[level1])*100:.1f}%)")
    if episodes_by_difficulty[level2] > 0:
        level_stats.append(f"Level {level2}: {solved_episodes[level2]}/{episodes_by_difficulty[level2]} "
                    f"({(solved_episodes[level2]/episodes_by_difficulty[level2])*100:.1f}%)")
    
    # Then all difficulties
    for d in sorted(all_solved_by_difficulty.keys()):
        if all_episodes_by_difficulty[d] > 0:
            level_stats.append(f"Difficulty {d}: {all_solved_by_difficulty[d]}/{all_episodes_by_difficulty[d]} "
                      f"({(all_solved_by_difficulty[d]/all_episodes_by_difficulty[d])*100:.1f}%)")
    
    # Calculate average loss for logging
    avg_loss = sum(training_losses[-100:]) / len(training_losses[-100:]) if training_losses else 0
    
    # Calculate average reward for logging
    avg_reward = sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0
    
    # Log training results to file
    log_file = log_training_results(
        scramble_moves=f"{level1}-{level2} mix",
        episode=episode,
        solved_episodes=solved_episodes[level2],  # Report higher level successes
        total_episodes=episodes_by_difficulty[level2],
        success_rate=success_rate_level2,
        recent_success_rate=recent_success_rate_level2,
        time_taken=total_time,
        agent_config=default_config,
        level_stats=level_stats,
        recent_scrambles=recent_scrambles,
        avg_reward=avg_reward,
        avg_loss=avg_loss,
        epsilon=agent.epsilon,
        memory_size=agent.get_memory_length(),
        best_success_rate=best_combined_rate,
        episodes_since_improvement=episodes_since_improvement,
        stopping_reason=stopping_reason
    )
    
    # Use the best model if we have one
    if os.path.exists(best_checkpoint) and best_combined_rate > combined_rate:
        print(f"Loading best model with combined success rate {best_combined_rate:.2f}% for testing")
        agent.policy_net.load_state_dict(torch.load(best_checkpoint))
        # Copy best model to final checkpoint
        torch.save(agent.policy_net.state_dict(), final_checkpoint)
    
    # Test on both difficulty levels
    print(f"\n=== Testing agent on level {level1} ===")
    test_agent(num_tests=30, scramble_moves=level1, checkpoint_path=final_checkpoint, use_pregenerated=use_pregenerated)
    
    print(f"\n=== Testing agent on level {level2} ===")
    test_agent(num_tests=30, scramble_moves=level2, checkpoint_path=final_checkpoint, use_pregenerated=use_pregenerated)
    
    return final_checkpoint

def progressive_training(start_level=None, max_scramble=20, min_episodes=5000, 
                         max_episodes=10000, target_success_rate=30, min_success_rate=None, batch_size=64,
                         use_pregenerated=True, custom_checkpoint=None, recent_window=1000, agent_config=None):
    """Train progressively with increasing scramble difficulty
    
    Args:
        start_level: Starting difficulty level (scramble moves). If None, continue from highest checkpoint.
        max_scramble: Maximum difficulty level to train up to
        min_episodes: Minimum episodes per difficulty level
        max_episodes: Maximum episodes per difficulty level
        target_success_rate: Target success rate to achieve before moving to next level
        min_success_rate: Minimum success rate to achieve regardless of episode count
        batch_size: Batch size for training
        use_pregenerated: Whether to use pregenerated scrambles
        custom_checkpoint: Path to a specific checkpoint to use instead of searching for latest
        recent_window: Number of recent episodes to consider for success rate calculation
        agent_config: Dictionary of DQNAgent configuration parameters
    """
    
    # If no start_level is specified, try to find the highest level with a checkpoint
    if start_level is None:
        start_level = find_latest_checkpoint_level() + 1
        if start_level > max_scramble:
            print(f"Already have checkpoints up to level {start_level-1}, which exceeds max_scramble={max_scramble}")
            return
        
    # Make sure we start at least from level 1
    start_level = max(1, start_level)
    print(f"Starting progressive training from level {start_level} up to {max_scramble}")
    
    checkpoint = custom_checkpoint if custom_checkpoint else None
    
    # If no custom checkpoint was provided and we're starting beyond level 1, 
    # load the checkpoint from the previous level
    if checkpoint is None and start_level > 1:
        prev_level = start_level - 1
        prev_checkpoint = os.path.join("modelCheckpoints", f'cube_solver_model_scramble_{prev_level}.pt')
        if os.path.exists(prev_checkpoint):
            checkpoint = prev_checkpoint
            print(f"Will use checkpoint from level {prev_level}: {prev_checkpoint}")
        else:
            print(f"Warning: Starting at level {start_level} but no checkpoint found for level {prev_level}")
    elif checkpoint:
        print(f"Using custom checkpoint: {checkpoint}")
    
    # Print agent configuration if provided
    if agent_config:
        print("Using custom agent configuration:")
        for key, value in agent_config.items():
            print(f"  - {key}: {value}")
            
    # Train progressively from start_level to max_scramble
    for scramble_moves in range(start_level, max_scramble + 1):
        # For each difficulty level, customize agent config if needed
        current_agent_config = agent_config.copy() if agent_config else {}
        
        # You can customize config per level if needed, for example:
        # if scramble_moves > 5:
        #    current_agent_config['epsilon_decay'] = 0.997  # Slower decay for harder levels
        
        checkpoint = train_specific_level(
            scramble_moves=scramble_moves,
            min_episodes=min_episodes,
            max_episodes=max_episodes,
            target_success_rate=target_success_rate,
            min_success_rate=min_success_rate,
            batch_size=batch_size,
            prev_checkpoint=checkpoint,
            use_pregenerated=use_pregenerated,
            recent_window=recent_window,
            agent_config=current_agent_config
        )

def test_agent(num_tests=100, scramble_moves=1, checkpoint_path=None, use_pregenerated=True):
    """Test a trained agent on cube solving"""
    # Ensure scrambles exist if using pregenerated scrambles
    if use_pregenerated:
        if not ensure_scrambles_exist(scramble_moves, use_pregenerated):
            print(f"Warning: Could not generate scrambles for testing level {scramble_moves}. Using random scrambles.")
            use_pregenerated = False
    
    env = CubeEnvironment(scramble_moves=scramble_moves, use_pregenerated=use_pregenerated)
    state_size = 6 * 9 * 6
    action_size = len(MOVES)
    
    # Initialize agent with default configuration for testing
    # For testing we don't need some features like prioritized replay
    agent = DQNAgent(
        state_size=state_size, 
        action_size=action_size,
        epsilon_start=0.0,  # No exploration during testing
        epsilon_min=0.0,
        prioritized_replay=False  # Not needed for testing
    )
    
    # Load checkpoint if provided
    if checkpoint_path:
        try:
            agent.policy_net.load_state_dict(torch.load(checkpoint_path))
            # Also load to target network (though we won't use it for testing)
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"Testing with checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            return
    
    # Set epsilon to minimum for best performance (minimal exploration)
    agent.epsilon = 0.0  # No exploration for testing
    
    solved_count = 0
    total_moves = 0
    
    # Track recent test results
    recent_tests = []
    
    # Track all solution moves for statistics
    all_solution_moves = []
    
    for test in range(num_tests):
        state = env.reset()
        done = False
        moves = 0
        
        # Get the scramble that was applied
        scramble = " ".join(env.scramble_sequence) if env.scramble_sequence else "Random scramble"
        
        while not done and moves < env.max_steps:
            action = agent.act(state)
            state, _, done = env.step(action)
            moves += 1
            
            if done and env.cube.is_solved():
                solved_count += 1
                total_moves += moves
                all_solution_moves.append(moves)
                recent_tests.append((test+1, scramble, "Solved", moves))
                break
        
        if not done or not env.cube.is_solved():
            recent_tests.append((test+1, scramble, "Failed", moves))
        
        # Keep only the most recent 10 tests
        if len(recent_tests) > 10:
            recent_tests.pop(0)
        
        # Update display every few tests
        if (test + 1) % 5 == 0 or test == 0 or test == num_tests - 1:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Display test progress
            print(f"=== Testing Progress === ")
            print(f"Testing Level: {scramble_moves}")
            print(f"Using Checkpoint: {checkpoint_path}")
            print(f"Test: {test+1}/{num_tests}")
            print()
            
            # Current statistics
            current_success_rate = (solved_count / (test+1)) * 100
            current_avg_moves = total_moves / solved_count if solved_count > 0 else 0
            
            print(f"Current Statistics:")
            print(f"✓ Success Rate: {current_success_rate:.2f}%")
            print(f"✓ Average Moves: {current_avg_moves:.2f}")
            print(f"✓ Solved: {solved_count}/{test+1}")
            
            # Calculate median and percentiles if we have solved cases
            if all_solution_moves:
                median_moves = sorted(all_solution_moves)[len(all_solution_moves)//2]
                p25_moves = sorted(all_solution_moves)[int(len(all_solution_moves)*0.25)]
                p75_moves = sorted(all_solution_moves)[int(len(all_solution_moves)*0.75)]
                min_moves = min(all_solution_moves)
                max_moves = max(all_solution_moves)
                print(f"✓ Move Statistics: min={min_moves}, 25%={p25_moves}, median={median_moves}, 75%={p75_moves}, max={max_moves}")
            
            print()
            
            # Recent test results
            print(f"Recent Test Results:")
            for test_num, scr, result, mv in recent_tests:
                print(f"- Test {test_num}: {scr} ({result} in {mv} moves)")
    
    # Final statistics
    success_rate = (solved_count / num_tests) * 100
    avg_moves = total_moves / solved_count if solved_count > 0 else 0
    
    print(f"\nFinal Test Results:")
    print(f"Scramble Moves: {scramble_moves}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Moves for Successful Solves: {avg_moves:.2f}")
    
    # More detailed move statistics
    if all_solution_moves:
        all_solution_moves.sort()
        median_moves = all_solution_moves[len(all_solution_moves)//2]
        p25_moves = all_solution_moves[int(len(all_solution_moves)*0.25)]
        p75_moves = all_solution_moves[int(len(all_solution_moves)*0.75)]
        min_moves = min(all_solution_moves)
        max_moves = max(all_solution_moves)
        print(f"Move Distribution: min={min_moves}, 25%={p25_moves}, median={median_moves}, 75%={p75_moves}, max={max_moves}")
    
    # Log test results to file
    log_file = log_training_results(
        scramble_moves=scramble_moves,
        episode=num_tests,
        solved_episodes=solved_count,
        total_episodes=num_tests,
        success_rate=success_rate,
        recent_success_rate=success_rate,
        time_taken=0,  # We don't track time for testing
        agent_config={"test_mode": True, "checkpoint": checkpoint_path},
        recent_scrambles=recent_tests
    )
    
    return success_rate, avg_moves

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Rubik\'s Cube solving agent using DQN')
    parser.add_argument('--level', type=int, default=None, 
                        help='Starting difficulty level (scramble moves). If not provided, will start from next level after highest checkpoint')
    parser.add_argument('--max_level', type=int, default=5, 
                        help='Maximum difficulty level to train up to')
    parser.add_argument('--min_episodes', type=int, default=5000, 
                        help='Minimum episodes per difficulty level')
    parser.add_argument('--max_episodes', type=int, default=10000, 
                        help='Maximum episodes per difficulty level')
    parser.add_argument('--target_rate', type=int, default=50, 
                        help='Target success rate to achieve before moving to next level')
    parser.add_argument('--min_rate', type=int, default=None, 
                        help='Minimum success rate to achieve regardless of episode count')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training')
    parser.add_argument('--use_pregenerated', action='store_true',
                        help='Use pregenerated scrambles from nMovescramble.txt instead of random scrambles')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a specific model checkpoint to use instead of searching for the latest checkpoint')
    parser.add_argument('--recent_window', type=int, default=1000,
                        help='Number of recent episodes to consider for calculating success rate')
    
    # DQN Agent configuration arguments
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the DQN agent')
    parser.add_argument('--memory_size', type=int, default=100000,
                        help='Memory size for experience replay')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for future rewards')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='Starting exploration rate')
    parser.add_argument('--epsilon_min', type=float, default=0.05,
                        help='Minimum exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                        help='Decay rate for exploration')
    parser.add_argument('--target_update', type=int, default=1000,
                        help='How often to update target network (steps)')
    
    # Advanced DQN features
    parser.add_argument('--no_double_dqn', action='store_false', dest='double_dqn',
                        help='Disable Double DQN (use regular DQN)')
    parser.add_argument('--no_dueling_dqn', action='store_false', dest='dueling_dqn',
                        help='Disable Dueling DQN architecture')
    parser.add_argument('--no_prioritized', action='store_false', dest='prioritized_replay',
                        help='Disable prioritized experience replay')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Priority exponent (alpha) for prioritized replay')
    parser.add_argument('--beta', type=float, default=0.4,
                        help='Initial importance sampling weight (beta) for prioritized replay')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run tests on the specified checkpoint')
    parser.add_argument('--test_level', type=int, default=None,
                        help='Difficulty level to test (required with --test_only)')
    parser.add_argument('--num_tests', type=int, default=100,
                        help='Number of test cases to run with --test_only')
    
    # Set defaults for advanced features
    parser.set_defaults(double_dqn=True, dueling_dqn=True, prioritized_replay=True)
    
    # Add new arguments for mixed training
    parser.add_argument('--mixed', action='store_true',
                        help='Train on a mix of two difficulty levels')
    parser.add_argument('--level1', type=int, default=6,
                        help='First difficulty level for mixed training (lower)')
    parser.add_argument('--level2', type=int, default=7,
                        help='Second difficulty level for mixed training (higher)')
    parser.add_argument('--mix_ratio', type=float, default=0.5,
                        help='Ratio of level2 vs level1 (0.5 means 50% level2, 50% level1)')
    
    args = parser.parse_args()
    
    # Build agent configuration
    agent_config = {
        'learning_rate': args.lr,
        'memory_size': args.memory_size,
        'gamma': args.gamma,
        'epsilon_start': args.epsilon_start,
        'epsilon_min': args.epsilon_min,
        'epsilon_decay': args.epsilon_decay,
        'target_update_freq': args.target_update,
        'double_dqn': args.double_dqn,
        'dueling_dqn': args.dueling_dqn,
        'prioritized_replay': args.prioritized_replay,
        'alpha': args.alpha,
        'beta': args.beta
    }
    
    # If test_only is specified, only run tests
    if args.test_only:
        if args.test_level is None:
            print("Error: --test_level must be specified with --test_only")
            sys.exit(1)
        if args.model is None:
            checkpoint = os.path.join("modelCheckpoints", f'cube_solver_model_scramble_{args.test_level}.pt')
            if not os.path.exists(checkpoint):
                print(f"Error: No checkpoint found for level {args.test_level}. Please specify --model.")
                sys.exit(1)
        else:
            checkpoint = args.model
            
        test_agent(
            num_tests=args.num_tests, 
            scramble_moves=args.test_level, 
            checkpoint_path=checkpoint, 
            use_pregenerated=args.use_pregenerated
        )
    elif args.mixed:
        # Build agent configuration
        agent_config = {
            'learning_rate': args.lr,
            'memory_size': args.memory_size,
            'gamma': args.gamma,
            'epsilon_start': args.epsilon_start,
            'epsilon_min': args.epsilon_min,
            'epsilon_decay': args.epsilon_decay,
            'target_update_freq': args.target_update,
            'double_dqn': args.double_dqn,
            'dueling_dqn': args.dueling_dqn,
            'prioritized_replay': args.prioritized_replay,
            'alpha': args.alpha,
            'beta': args.beta
        }
        
        # Run mixed training
        train_mixed_levels(
            level1=args.level1,
            level2=args.level2,
            mix_ratio=args.mix_ratio,
            min_episodes=args.min_episodes,
            max_episodes=args.max_episodes,
            target_success_rate=args.target_rate,
            min_success_rate=args.min_rate,
            batch_size=args.batch_size,
            use_pregenerated=args.use_pregenerated,
            custom_checkpoint=args.model,
            recent_window=args.recent_window,
            agent_config=agent_config
        )
    else:
        # Run progressive training with specified or default parameters
        progressive_training(
            start_level=args.level,
            max_scramble=args.max_level,
            min_episodes=args.min_episodes,
            max_episodes=args.max_episodes,
            target_success_rate=args.target_rate,
            min_success_rate=args.min_rate,
            batch_size=args.batch_size,
            use_pregenerated=args.use_pregenerated,
            custom_checkpoint=args.model,
            recent_window=args.recent_window,
            agent_config=agent_config
        )
    
    print("\n=== Examples for training and testing ===")
    print("# Train level 4 from scratch with optimized settings:")
    print("python cube_rl.py --level 4 --max_level 4 --lr 0.001 --epsilon_decay 0.995 --memory_size 150000 --min_episodes 8000 --max_episodes 20000 --target_rate 60 --use_pregenerated")
    
    print("\n# Continue training level 4 from an existing checkpoint:")
    print("python cube_rl.py --level 4 --max_level 4 --model cube_solver_model_scramble_4.pt --lr 0.0005 --epsilon_start 0.2 --min_episodes 5000 --target_rate 65 --use_pregenerated")
    
    print("\n# Test a trained model on level 4:")
    print("python cube_rl.py --test_only --test_level 4 --num_tests 200 --use_pregenerated")
    
    print("\n# Progressive training from level 1 to 10:")
    print("python cube_rl.py --max_level 10 --min_episodes 5000 --target_rate 50 --use_pregenerated --memory_size 200000")
    
    print("\n# Train with mixed difficulty (level 6 and 7 with 50% mix):")
    print("python cube_rl.py --mixed --level1 6 --level2 7 --mix_ratio 0.5 --min_episodes 10000 --max_episodes 30000 --target_rate 5 --use_pregenerated")
    
    # To test the latest checkpoint:
    # test_agent(num_tests=100, scramble_moves=5, checkpoint_path="cube_solver_model_scramble_5.pt", use_pregenerated=True) 
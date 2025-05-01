import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from cube import Cube
import kociemba as koc
import os
import argparse
import sys
import helper

# Create modelCheckpoints directory if it doesn't exist
os.makedirs("data/modelCheckpoints", exist_ok=True)

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
            helper.load_scrambles(self, scramble_moves)
            
    def reset(self):
        self.cube = Cube()
        self.current_step = 0
        self.agent_moves = []  # Reset the moves list
        
        # Check if we need to regenerate scrambles
        helper.check_and_regenerate_scrambles(self)
        
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
        
        if len(self.agent_moves) >= 2:
            if self.agent_moves[-1] == self.agent_moves[-2]:
                penalties.append(-15)  # Increased penalty for two consecutive identical moves
                
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
                
        # --- NEW: DETECT AND PENALIZE LARGER REPETITIVE PATTERNS ---
        # Check for repeating patterns of length 2 (like "L R L R L R")
        if len(self.agent_moves) >= 6:
            # Check if the last 6 moves form a repeating 2-move pattern
            if (self.agent_moves[-6] == self.agent_moves[-4] == self.agent_moves[-2] and
                self.agent_moves[-5] == self.agent_moves[-3] == self.agent_moves[-1]):
                penalties.append(-25)  # Strong penalty for 3 repetitions of a 2-move sequence
                
        # Check for repeating patterns of length 3 (like "L R F L R F")
        if len(self.agent_moves) >= 9:
            # Check if the last 9 moves form a repeating 3-move pattern
            if (self.agent_moves[-9] == self.agent_moves[-6] == self.agent_moves[-3] and
                self.agent_moves[-8] == self.agent_moves[-5] == self.agent_moves[-2] and
                self.agent_moves[-7] == self.agent_moves[-4] == self.agent_moves[-1]):
                penalties.append(-30)  # Stronger penalty for 3 repetitions of a 3-move sequence
                
        # Penalize long stretches of the same face being manipulated
        # (like "D D' D2 D D' D" - all on the D face)
        if len(self.agent_moves) >= 5:
            last_five_faces = [move[0] for move in self.agent_moves[-5:]]
            if len(set(last_five_faces)) == 1:  # All moves were on the same face
                penalties.append(-20)  # Penalty for obsessing over a single face
                
        # Penalize oscillating between just two faces for many moves
        if len(self.agent_moves) >= 8:
            last_eight_faces = [move[0] for move in self.agent_moves[-8:]]
            unique_faces = set(last_eight_faces)
            if len(unique_faces) == 2 and all(face in unique_faces for face in last_eight_faces):
                penalties.append(-25)  # Penalty for oscillating between just two faces
        
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
    
    # New argument for curriculum training
    parser.add_argument('--curriculum', action='store_true',
                        help='Use continuous curriculum training instead of progressive training')
    parser.add_argument('--success_threshold', type=int, default=60,
                        help='Success rate threshold to unlock next difficulty level in curriculum training')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='How often to save checkpoints during curriculum training (in episodes)')
    parser.add_argument('--plateau_patience', type=int, default=500000,
                        help='Number of episodes to wait before advancing level if no improvement')
    # Set defaults for advanced features
    parser.set_defaults(double_dqn=True, dueling_dqn=True, prioritized_replay=True)
    
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
            checkpoint = os.path.join("data/modelCheckpoints", f'cube_solver_model_scramble_{args.test_level}.pt')
            if not os.path.exists(checkpoint):
                print(f"Error: No checkpoint found for level {args.test_level}. Please specify --model.")
                sys.exit(1)
        else:
            checkpoint = args.model
            
        helper.test_agent(
            num_tests=args.num_tests, 
            scramble_moves=args.test_level, 
            checkpoint_path=checkpoint, 
            use_pregenerated=args.use_pregenerated
        )
    else:
        # Choose between curriculum training and progressive training
        if args.curriculum:
            print("Using continuous curriculum training mode")
            helper.continuous_curriculum_training(
                max_scramble=args.max_level,
                min_episodes=args.min_episodes,
                max_episodes=args.max_episodes,
                success_threshold=args.success_threshold,
                batch_size=args.batch_size,
                checkpoint_path=args.model,
                use_pregenerated=args.use_pregenerated,
                checkpoint_interval=args.checkpoint_interval,
                recent_window=args.recent_window,
                agent_config=agent_config,
                plateau_patience=args.plateau_patience,
            )
        else:
            # Run progressive training with specified or default parameters
            helper.progressive_training(
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
    
    print("\n# Curriculum training from level 1 to 10:")
    print("python cube_rl.py --curriculum --max_level 10 --min_episodes 50000 --max_episodes 200000 --success_threshold 60 --use_pregenerated --memory_size 150000")
    
    # To test the latest checkpoint:
    # test_agent(num_tests=100, scramble_moves=5, checkpoint_path="cube_solver_model_scramble_5.pt", use_pregenerated=True) 
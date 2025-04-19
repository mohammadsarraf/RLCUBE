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
        
    def reset(self):
        self.cube = Cube()
        self.current_step = 0
        self.agent_moves = []  # Reset the moves list
        
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
        
        # Base reward for solving the cube
        if solved:
            reward += 100
        else:
            reward -= 10  # Reduced negative reward, was too harsh at -100
        
        # Penalize repeating moves (like R R R instead of R')
        if len(self.agent_moves) >= 3:
            last_three_moves = self.agent_moves[-3:]
            if last_three_moves[0] == last_three_moves[1] == last_three_moves[2]:
                reward -= 10  # Penalty for three consecutive identical moves
        
        # Penalize sequences like R2 R or R2 R' (inefficient move combinations)
        if len(self.agent_moves) >= 2:
            last_move = self.agent_moves[-1]
            prev_move = self.agent_moves[-2]
            
            # Check if previous move was a double move (R2) and current move is on the same face (R or R')
            if "2" in prev_move and not "2" in last_move:
                # If they're moves on the same face
                if prev_move.replace("2", "") == last_move.replace("'", ""):
                    reward -= 10  # Penalty for inefficient sequence
            
            # Check for the reverse case: R followed by R2 (another inefficient sequence)
            elif "2" in last_move and not "2" in prev_move:
                # If they're moves on the same face
                if last_move.replace("2", "") == prev_move.replace("'", ""):
                    reward -= 10  # Penalty for inefficient sequence
        
        # Penalize inverse moves that cancel each other (like R R')
        if len(self.agent_moves) >= 2:
            last_move = self.agent_moves[-1]
            prev_move = self.agent_moves[-2]
            
            # Check for move cancellations (simplified check)
            # For basic moves like R and R'
            if (last_move.replace("'", "") == prev_move.replace("'", "") and 
                ("'" in last_move) != ("'" in prev_move) and 
                "2" not in last_move and "2" not in prev_move):
                reward -= 10  # Larger penalty for canceling moves
        
        # Penalize inefficient sequences like B2 F B2 (where B2 B2 would cancel out)
        if len(self.agent_moves) >= 3:
            last_move = self.agent_moves[-1]
            second_last_move = self.agent_moves[-2]
            third_last_move = self.agent_moves[-3]
            
            # Check for patterns like X2 Y X2 where X2 cancels out
            if "2" in last_move and "2" in third_last_move:
                # If they're the same face (like U2 F U2)
                if last_move.replace("2", "") == third_last_move.replace("2", ""):
                    reward -= 10  # Penalty for inefficient sequences
            
            # Check for sequences where moves on opposite faces interact inefficiently
            # For example B2 F B2 is just F
            last_face = last_move[0]  # Get the face of the last move
            third_last_face = third_last_move[0]  # Get the face of the third-last move
            
            # If the moves are on opposite faces and both are double moves (2)
            if "2" in last_move and "2" in third_last_move:
                if self.opposite_faces.get(last_face) == third_last_face:
                    reward -= 10  # Penalty for inefficient sequences
            
            # Check for patterns like L R L' where a move is followed by another move
            # and then the inverse of the first move
            if ("2" not in third_last_move and 
                third_last_move.replace("'", "") != second_last_move.replace("'", "") and 
                third_last_move.replace("'", "") == last_move.replace("'", "") and
                ("'" in third_last_move) != ("'" in last_move)):
                reward -= 10  # Penalty for inefficient sequences like L R L'
        
        # Use kociemba to estimate the distance to the solution
        try:
            solution = koc.solve(self.cube.to_kociemba_string())
            solution_length = len(solution.split())
            
            # Negative reward based on solution length (shorter is better)
            reward -= 1 * solution_length  # Reduced from 1 to 0.5
            
            # Extra reward for solutions shorter than Kociemba's
            agent_solution_length = len(self.agent_moves)
            if solved and agent_solution_length < solution_length:
                reward += (solution_length - agent_solution_length) * 5
            
            # Large reward for solutions under 20 moves (God's number)
            if solved and solution_length <= 20:
                reward += 200 - (solution_length * 5)  # Higher reward for shorter solutions
        
        except:
            # If kociemba fails (invalid cube state), give a small negative reward
            reward -= 1
        
        return reward

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)  # Increase memory size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05  # Slightly higher minimum epsilon
        self.epsilon_decay = 0.998  # Slower decay
        self.learning_rate = 0.001
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state_tensor)
        self.model.train()
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            
            # Get current Q value
            self.model.eval()
            with torch.no_grad():
                current_q = self.model(state_tensor)
            self.model.train()
            
            # Update the Q value for the action
            target_f = current_q.clone()
            target_f[0][action] = target
            
            # Train
            self.optimizer.zero_grad()
            outputs = self.model(state_tensor)
            loss = self.criterion(outputs, target_f)
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def find_latest_checkpoint_level():
    """Find the highest difficulty level with an existing checkpoint"""
    max_level = 0
    for file in os.listdir():
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

def train_specific_level(scramble_moves, min_episodes=5000, max_episodes=50000, 
                         target_success_rate=30, batch_size=64, prev_checkpoint=None, 
                         use_pregenerated=False):
    """Train on a specific difficulty level"""
    print(f"\n=== Starting training with {scramble_moves} scramble moves ===")
    
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
    agent = DQNAgent(state_size, action_size)
    
    # Load previous checkpoint if available
    if prev_checkpoint:
        try:
            agent.model.load_state_dict(torch.load(prev_checkpoint))
            print(f"Loaded checkpoint from {prev_checkpoint}")
            # Reset epsilon to a higher value when moving to a new difficulty
            agent.epsilon = max(0.5, agent.epsilon)  # Ensure at least 50% exploration for new difficulty
        except:
            print(f"Failed to load checkpoint from {prev_checkpoint}, starting fresh")
    
    solved_episodes = 0
    solved_by_difficulty = {i: 0 for i in range(1, scramble_moves + 1)}
    episodes_by_difficulty = {i: 0 for i in range(1, scramble_moves + 1)}
    start_time = time.time()
    current_success_rate = 0
    episode = 0
    
    # Keep track of recent scrambles and outcomes
    recent_scrambles = []  # Will store tuples of (scramble, outcome, moves)
    
    # Define the final checkpoint name for this difficulty level
    final_checkpoint = f'cube_solver_model_scramble_{scramble_moves}.pt'
    
    # Training loop - continue until we reach target success rate or max episodes
    while episode < max_episodes:
        episode += 1
        
        # Distribution of difficulties:
        # 70% current difficulty, 30% distributed among easier difficulties
        if scramble_moves == 1 or random.random() < 0.7:
            # Use current difficulty most of the time
            selected_diff = scramble_moves
        else:
            # Select from easier difficulties with preference toward harder ones
            weights = [i/sum(range(1, scramble_moves)) for i in range(1, scramble_moves)]
            selected_diff = random.choices(range(1, scramble_moves), weights=weights)[0]
        
        # Use the environment with selected difficulty
        env_to_use = all_envs[selected_diff]
        episodes_by_difficulty[selected_diff] += 1
        
        state = env_to_use.reset()
        done = False
        moves_taken = 0
        
        # Get the scramble that was applied
        scramble = " ".join(env_to_use.agent_moves) if env_to_use.agent_moves else "Random scramble"
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env_to_use.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            moves_taken += 1
            
            if done:
                outcome = "Solved" if env_to_use.cube.is_solved() else "Failed"
                # Add to recent scrambles
                recent_scrambles.append((scramble, outcome, moves_taken))
                # Keep only the 5 most recent scrambles
                if len(recent_scrambles) > 5:
                    recent_scrambles.pop(0)
                
                if outcome == "Solved":
                    solved_by_difficulty[selected_diff] += 1
                    if selected_diff == scramble_moves:  # Count solutions for current difficulty
                        solved_episodes += 1
                break
        
        # Train the agent with experiences from memory
        agent.replay(batch_size)
        
        # Print progress and check success rate every 10 episodes
        if episode % 100 == 0 or episode == 1:
            # Calculate success rate for current difficulty
            main_episodes = episodes_by_difficulty[scramble_moves]
            main_solved = solved_by_difficulty[scramble_moves]
            
            if main_episodes > 0:
                current_success_rate = (main_solved / main_episodes) * 100
            
            all_solved = sum(solved_by_difficulty.values())
            all_episodes = sum(episodes_by_difficulty.values())
            overall_rate = (all_solved / all_episodes) * 100
            
            # Clear screen for cleaner display
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Display training progress
            print(f"=== Training Progress === - Max Level: {scramble_moves} - Min Episodes: {min_episodes} - Max Episodes: {max_episodes} - Target Success Rate: {target_success_rate}% - Batch Size: {batch_size} - Using Pregenerated Scrambles: {use_pregenerated} - Previous Checkpoint: {prev_checkpoint if prev_checkpoint else 'None'}")
            print(f"  - Using Pregenerated Scrambles: {use_pregenerated}")
            print(f"  - Previous Checkpoint: {prev_checkpoint if prev_checkpoint else 'None'}")
            print("=" * 50)
            print(f"Level: {scramble_moves}")
            print(f"Episode: {episode}/{max_episodes}")
            print()
            
            # Performance metrics
            print(f"Performance Summary:")
            print(f"✓ Solved Rate (Current Level): {current_success_rate:.2f}%")
            print(f"✓ Overall Success Rate: {overall_rate:.2f}%")
            print(f"✓ Exploration Rate (Epsilon): {agent.epsilon:.4f}")
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
            for i, (scr, out, mvs) in enumerate(recent_scrambles):
                print(f"- {scr} ({out} in {mvs} moves)")
            
            # Save intermediate checkpoint, replacing the previous one
            if episode % 1000 == 0:
                torch.save(agent.model.state_dict(), final_checkpoint)
                print(f"\nSaved checkpoint: {final_checkpoint}")
        
        # Check if we've met the success criteria and minimum episodes
        if episode >= min_episodes and current_success_rate >= target_success_rate:
            print(f"\nReached target success rate of {target_success_rate}% after {episode} episodes!")
            break
    
    # Save final model for this scramble difficulty
    torch.save(agent.model.state_dict(), final_checkpoint)
    total_time = time.time() - start_time
    print(f"\nTraining completed. Scramble moves: {scramble_moves}, "
          f"Episodes: {episode}, Solved: {solved_episodes}/{episodes_by_difficulty[scramble_moves]}, "
          f"Success Rate: {current_success_rate:.2f}%, Time: {total_time:.2f} seconds.")
    
    # Test on current difficulty
    print(f"\n=== Testing agent with {scramble_moves} scramble moves ===")
    test_agent(num_tests=50, scramble_moves=scramble_moves, checkpoint_path=final_checkpoint, use_pregenerated=use_pregenerated)
    
    return final_checkpoint

def progressive_training(start_level=None, max_scramble=20, min_episodes=5000, 
                         max_episodes=50000, target_success_rate=30, batch_size=64,
                         use_pregenerated=True):
    """Train progressively with increasing scramble difficulty"""
    
    # If no start_level is specified, try to find the highest level with a checkpoint
    if start_level is None:
        start_level = find_latest_checkpoint_level() + 1
        if start_level > max_scramble:
            print(f"Already have checkpoints up to level {start_level-1}, which exceeds max_scramble={max_scramble}")
            return
        
    # Make sure we start at least from level 1
    start_level = max(1, start_level)
    print(f"Starting progressive training from level {start_level} up to {max_scramble}")
    
    checkpoint = None
    # If we're starting beyond level 1, load the checkpoint from the previous level
    if start_level > 1:
        prev_level = start_level - 1
        prev_checkpoint = f'cube_solver_model_scramble_{prev_level}.pt'
        if os.path.exists(prev_checkpoint):
            checkpoint = prev_checkpoint
            print(f"Will use checkpoint from level {prev_level}: {prev_checkpoint}")
        else:
            print(f"Warning: Starting at level {start_level} but no checkpoint found for level {prev_level}")
            
    # Train progressively from start_level to max_scramble
    for scramble_moves in range(start_level, max_scramble + 1):
        checkpoint = train_specific_level(
            scramble_moves=scramble_moves,
            min_episodes=min_episodes,
            max_episodes=max_episodes,
            target_success_rate=target_success_rate,
            batch_size=batch_size,
            prev_checkpoint=checkpoint,
            use_pregenerated=use_pregenerated
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
    agent = DQNAgent(state_size, action_size)
    
    # Load checkpoint if provided
    if checkpoint_path:
        try:
            agent.model.load_state_dict(torch.load(checkpoint_path))
            print(f"Testing with checkpoint: {checkpoint_path}")
        except:
            print(f"Failed to load checkpoint from {checkpoint_path}")
            return
    
    # Set epsilon to minimum for best performance (minimal exploration)
    agent.epsilon = agent.epsilon_min
    
    solved_count = 0
    total_moves = 0
    
    # Track recent test results
    recent_tests = []
    
    for test in range(num_tests):
        state = env.reset()
        done = False
        moves = 0
        
        # Get the scramble that was applied
        scramble = " ".join(env.agent_moves) if env.agent_moves else "Random scramble"
        
        while not done and moves < env.max_steps:
            action = agent.act(state)
            state, _, done = env.step(action)
            moves += 1
            
            if done and env.cube.is_solved():
                solved_count += 1
                total_moves += moves
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
            print(f"Test: {test+1}/{num_tests}")
            print()
            
            # Current statistics
            current_success_rate = (solved_count / (test+1)) * 100
            current_avg_moves = total_moves / solved_count if solved_count > 0 else 0
            
            print(f"Current Statistics:")
            print(f"✓ Success Rate: {current_success_rate:.2f}%")
            print(f"✓ Average Moves: {current_avg_moves:.2f}")
            print(f"✓ Solved: {solved_count}/{test+1}")
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
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Moves for Successful Solves: {avg_moves:.2f}")
    
    return success_rate, avg_moves

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Rubik\'s Cube solving agent using DQN')
    parser.add_argument('--level', type=int, default=None, 
                        help='Starting difficulty level (scramble moves). If not provided, will start from next level after highest checkpoint')
    parser.add_argument('--max_level', type=int, default=5, 
                        help='Maximum difficulty level to train up to')
    parser.add_argument('--min_episodes', type=int, default=5000, 
                        help='Minimum episodes per difficulty level')
    parser.add_argument('--max_episodes', type=int, default=50000, 
                        help='Maximum episodes per difficulty level')
    parser.add_argument('--target_rate', type=int, default=30, 
                        help='Target success rate to achieve before moving to next level')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training')
    parser.add_argument('--use_pregenerated', action='store_true',
                        help='Use pregenerated scrambles from nMovescramble.txt instead of random scrambles')
    
    args = parser.parse_args()
    
    # Run progressive training with specified or default parameters
    progressive_training(
        start_level=args.level,
        max_scramble=args.max_level,
        min_episodes=args.min_episodes,
        max_episodes=args.max_episodes,
        target_success_rate=args.target_rate,
        batch_size=args.batch_size,
        use_pregenerated=args.use_pregenerated
    )
    
    # To test the latest checkpoint:
    # test_agent(num_tests=100, scramble_moves=5, checkpoint_path="cube_solver_model_scramble_5.pt", use_pregenerated=True) 
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from cube import Cube
import kociemba as koc
import time  # Add time module for timing

# Define possible moves
MOVES = ["U", "U'", "U2", "D", "D'", "D2", "L", "L'", "L2", "R", "R'", "R2", "F", "F'", "F2", "B", "B'", "B2"]

class CubeEnvironment:
    def __init__(self, max_steps=25):
        self.cube = Cube()
        self.max_steps = max_steps  # Solution shouldn't be more than 25 moves
        self.current_step = 0
        self.agent_moves = []  # Track the sequence of moves
        
    def reset(self):
        self.cube = Cube()
        # Apply just one move as requested
        move = random.choice(MOVES)
        self.cube.apply_algorithm(move)
        self.current_step = 0
        self.agent_moves = []  # Reset the moves list
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
        
        # Penalize repeating moves (like R R R instead of R')
        if len(self.agent_moves) >= 3:
            last_three_moves = self.agent_moves[-3:]
            if last_three_moves[0] == last_three_moves[1] == last_three_moves[2]:
                reward -= 5  # Penalty for three consecutive identical moves
        
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
        
        # Use kociemba to estimate the distance to the solution
        try:
            solution = koc.solve(self.cube.to_kociemba_string())
            solution_length = len(solution.split())
            
            # Negative reward based on solution length (shorter is better)
            reward -= 1 * solution_length
            
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
        

        # Reward for making progress (completing faces)
        # This is a simplified example - you'd need to implement face completion detection
        # completed_faces = self._count_completed_faces()
        # reward += completed_faces * 10
        
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
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
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

def train_agent(episodes, batch_size=64):
    env = CubeEnvironment()
    state_size = 6 * 9 * 6  # 6 faces, 9 stickers per face, 6 possible colors
    action_size = len(MOVES)
    agent = DQNAgent(state_size, action_size)
    
    solved_episodes = 0
    start_time = time.time()  # Start timer for total training time
    
    for episode in range(episodes):
        episode_start_time = time.time()  # Start timer for this episode
        state = env.reset()
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done and env.cube.is_solved():
                solved_episodes += 1
                episode_time = time.time() - episode_start_time
                # print(f"Episode: {episode+1}/{episodes}, Solved: {solved_episodes}, Epsilon: {agent.epsilon:.4f}, Time: {episode_time:.2f}s")
                break
        
        # Train the agent with experiences from memory
        agent.replay(batch_size)
        
        # Print progress
        if (episode+1) % 100 == 0:
            # episode_time = time.time() - episode_start_time
            # elapsed_time = time.time() - start_time
            solved_rate = (solved_episodes / (episode + 1)) * 100
            print(f"Episode: {episode+1}/{episodes}, Solved: {solved_episodes}, Solved Rate: {solved_rate:.2f}%, Epsilon: {agent.epsilon:.4f}")
    # Save trained model
    total_time = time.time() - start_time
    torch.save(agent.model.state_dict(), 'cube_solver_model.pt')
    print(f"Training completed. Solved {solved_episodes}/{episodes} episodes in {total_time:.2f} seconds.")
    return agent

if __name__ == "__main__":
    train_agent(episodes=1000) 
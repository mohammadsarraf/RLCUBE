import torch
import numpy as np
import random
from cube import Cube
from cube_rl import CubeEnvironment, MOVES, DQN

def test_agent(model_path='cube_solver_model.pt', num_tests=100):
    """
    Test the trained RL agent on cubes with one move applied.
    
    Args:
        model_path: Path to the saved model file
        num_tests: Number of test cases to run
    
    Returns:
        Success rate (percentage of cubes solved)
    """
    # Initialize environment
    env = CubeEnvironment(max_steps=50)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with the same architecture as in training
    state_size = 6 * 9 * 6  # 6 faces, 9 stickers per face, 6 possible colors
    action_size = len(MOVES)
    model = DQN(state_size, action_size).to(device)
    
    # Load the trained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    
    solved = 0
    avg_steps = 0
    
    for i in range(num_tests):
        # Reset the environment (applies one random move)
        state = env.reset()
        done = False
        steps = 0
        
        # Keep track of the initial move for reporting
        initial_state = env.cube.to_kociemba_string()
        
        # Track the solution moves
        solution_moves = []
        
        # Run until solved or max steps reached
        while not done:
            # Get action from model
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()
            
            # Record the move
            solution_moves.append(MOVES[action])
            
            # Apply action
            state, reward, done = env.step(action)
            steps += 1
            
            # Check if solved
            if done and env.cube.is_solved():
                solved += 1
                avg_steps += steps
                print(f"Test {i+1}: Solved in {steps} steps")
                print(f"Scramble: {initial_state}")
                print(f"Solution: {' '.join(solution_moves)}")
                break
            
            # Check if max steps reached without solving
            if done:
                print(f"Test {i+1}: Failed to solve in {steps} steps")
                print(f"Scramble: {initial_state}")
                print(f"Attempted solution: {' '.join(solution_moves)}")
                break
    
    # Calculate statistics
    success_rate = (solved / num_tests) * 100
    avg_steps = avg_steps / solved if solved > 0 else 0
    
    print(f"\nResults:")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average steps to solve (when solved): {avg_steps:.2f}")
    
    return success_rate

if __name__ == "__main__":
    try:
        test_agent()
    except FileNotFoundError:
        print("Model file not found. Please train the model first using cube_rl.py.") 
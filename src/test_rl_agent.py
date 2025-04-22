import torch
import numpy as np
import random
import os
import json
from cube import Cube
from cube_rl import CubeEnvironment, MOVES, DQN

def find_latest_checkpoint():
    """Find the latest checkpoint file based on scramble difficulty"""
    checkpoints = []
    # Look in modelCheckpoints directory instead of current directory
    if not os.path.exists("modelCheckpoints"):
        os.makedirs("modelCheckpoints", exist_ok=True)
        print("Created modelCheckpoints directory")
        return None
        
    for file in os.listdir("modelCheckpoints"):
        if file.startswith('cube_solver_model_scramble_') and file.endswith('.pt'):
            try:
                # Extract scramble level from filename
                scramble_level = int(file.split('_')[-1].split('.')[0])
                checkpoints.append((scramble_level, file))
            except ValueError:
                continue
    
    if not checkpoints:
        return None
    
    # Sort by scramble level (descending) to get the most advanced model
    checkpoints.sort(reverse=True)
    latest_checkpoint = os.path.join("modelCheckpoints", checkpoints[0][1])
    print(f"Found latest checkpoint for {checkpoints[0][0]} scramble moves: {latest_checkpoint}")
    return latest_checkpoint

def load_scrambles_from_file(n_moves, num_scrambles=100):
    """
    Load scrambles from the pregenerated file for n moves.
    
    Args:
        n_moves: Number of moves in the solution
        num_scrambles: Number of scrambles to load
        
    Returns:
        List of scramble data dictionaries
    """
    filepath = os.path.join("scrambles", f"{n_moves}movescramble.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scramble file not found: {filepath}. Generate it using gen.py --level {n_moves}")
    
    scrambles = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    scramble_data = json.loads(line.strip())
                    scrambles.append(scramble_data)
                    if len(scrambles) >= num_scrambles:
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error loading scrambles: {e}")
        return []
    
    if not scrambles:
        print(f"No valid scrambles found in {filepath}")
    else:
        print(f"Loaded {len(scrambles)} scrambles with {n_moves} solution length")
    
    return scrambles

def test_agent(model_path=None, scramble_moves=1, num_tests=100, use_pregenerated=False):
    """
    Test the trained RL agent on cubes with specified number of scramble moves.
    
    Args:
        model_path: Path to the saved model file (if None, will find latest checkpoint)
        scramble_moves: Number of moves to scramble the cube with
        num_tests: Number of test cases to run
        use_pregenerated: Whether to use pregenerated scrambles from file
    
    Returns:
        Success rate (percentage of cubes solved)
    """
    # Clear screen for cleaner display
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Find latest checkpoint if not specified
    if model_path is None:
        model_path = find_latest_checkpoint()
        if model_path is None:
            raise FileNotFoundError("No checkpoint files found. Please train the model first.")
        print(f"Using latest checkpoint: {model_path}")
    
    # Load pregenerated scrambles if requested
    pregenerated_scrambles = []
    if use_pregenerated:
        pregenerated_scrambles = load_scrambles_from_file(scramble_moves, num_tests)
        if not pregenerated_scrambles:
            print(f"Could not load pregenerated scrambles for {scramble_moves} moves. Using random scrambles.")
            use_pregenerated = False
    
    # Initialize environment with the specified number of scramble moves
    env = CubeEnvironment(max_steps=25, scramble_moves=scramble_moves)
    
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
    
    # Store recent test results
    recent_tests = []
    
    # Track all failed solutions
    failed_solutions = []
    
    print(f"=== Testing {num_tests} cubes with {scramble_moves} move scrambles ===")
    print(f"Using {'pregenerated' if use_pregenerated else 'random'} scrambles")
    
    for i in range(num_tests):
        # Reset the environment or apply a pregenerated scramble
        if use_pregenerated and i < len(pregenerated_scrambles):
            # Use a pregenerated scramble
            cube = Cube()
            scramble_data = pregenerated_scrambles[i]
            scramble = scramble_data["scramble"]
            expected_solution = scramble_data["solution"] if "solution" in scramble_data else "Unknown"
            
            # Apply the scramble
            cube.apply_algorithm(scramble)
            
            # Set up environment with scrambled cube
            env.cube = cube
            env.current_step = 0
            env.agent_moves = []
            state = env._get_state()
        else:
            # Use random scrambles from environment
            state = env.reset()
            # Get the actual scramble sequence that was applied
            scramble_sequence = env.scramble_sequence if hasattr(env, 'scramble_sequence') else []
            scramble = " ".join(scramble_sequence) if scramble_sequence else "No scramble recorded"
            expected_solution = "Unknown"
        
        # Keep track of the scramble for reporting
        initial_state = env.cube.to_kociemba_string()
        
        # Track the solution moves
        solution_moves = []
        done = False
        steps = 0
        
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
                test_result = {"test_num": i+1, "scramble": scramble, "outcome": "Solved", "steps": steps, 
                              "solution": " ".join(solution_moves), "expected": expected_solution}
                recent_tests.append(test_result)
                break
            
            # Check if max steps reached without solving
            if done:
                test_result = {"test_num": i+1, "scramble": scramble, "outcome": "Failed", "steps": steps, 
                              "solution": " ".join(solution_moves), "expected": expected_solution}
                recent_tests.append(test_result)
                
                # Add to failed solutions list
                failed_solutions.append({
                    "test_num": i+1,
                    "scramble": scramble,
                    "solution_attempt": " ".join(solution_moves),
                    "steps": steps
                })
                break
        
        # Keep only the last 10 tests for display
        if len(recent_tests) > 10:
            recent_tests.pop(0)
        
        # Update display every 5 tests
        if (i + 1) % 5 == 0 or i == 0 or i == num_tests - 1:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Display progress
            print(f"=== Testing Progress ===")
            print(f"Testing {scramble_moves}-move scrambles with {model_path}")
            print(f"Test: {i+1}/{num_tests}")
            print()
            
            # Current statistics
            current_success_rate = (solved / (i+1)) * 100
            current_avg_steps = avg_steps / solved if solved > 0 else 0
            
            print(f"Current Statistics:")
            print(f"✓ Success Rate: {current_success_rate:.2f}%")
            print(f"✓ Average Steps: {current_avg_steps:.2f}")
            print(f"✓ Solved: {solved}/{i+1}")
            print()
            
            # Recent test results
            print(f"Recent Test Results:")
            for result in recent_tests:
                print(f"- Test {result['test_num']}: {result['scramble']} ({result['outcome']} in {result['steps']} steps)")
                if result['outcome'] == 'Solved' and result['expected'] != 'Unknown':
                    print(f"  Expected solution: {result['expected']}")
                    print(f"  Agent solution: {result['solution']}")
    
    # Calculate final statistics
    success_rate = (solved / num_tests) * 100
    avg_steps = avg_steps / solved if solved > 0 else 0
    
    # Print failed solutions if any
    if failed_solutions:
        print(f"\nFailed Solutions ({len(failed_solutions)}):")
        for idx, failed in enumerate(failed_solutions):
            print(f"\n#{idx+1}. Test {failed['test_num']}")
            print(f"  Scramble: {failed['scramble']}")
            print(f"  Solution attempt: {failed['solution_attempt']}")
            print(f"  Steps: {failed['steps']}")
    

    print(f"\nFinal Results:")
    print(f"Scramble moves: {scramble_moves}")
    print(f"Model used: {model_path}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average steps to solve (when solved): {avg_steps:.2f}")
    

    return success_rate

def solve_manual_scramble(scramble_sequence, model_path=None, max_steps=25):
    """
    Solve a cube with a manually provided scramble sequence.
    
    Args:
        scramble_sequence: String of moves separated by spaces (e.g., "R U F' L2")
        model_path: Path to the model checkpoint to use
        max_steps: Maximum number of steps to attempt
        
    Returns:
        True if solved, False otherwise, and the solution moves
    """
    # Clear screen for cleaner display
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Find latest checkpoint if not specified
    if model_path is None:
        model_path = find_latest_checkpoint()
        if model_path is None:
            raise FileNotFoundError("No checkpoint files found. Please train the model first.")
        print(f"Using latest checkpoint: {model_path}")
    
    # Create a fresh cube
    cube = Cube()
    
    # Apply the scramble sequence
    scramble_moves = scramble_sequence.split()
    print(f"Applying scramble: {scramble_sequence}")
    for move in scramble_moves:
        cube.apply_algorithm(move)
    
    # Check if scramble is valid (sometimes input can be malformed)
    try:
        kociemba_state = cube.to_kociemba_string()
        print(f"Scrambled cube state: {kociemba_state}")
    except:
        print("Warning: The scramble produced an invalid cube state.")
    
    # Set up the environment with the scrambled cube
    env = CubeEnvironment(max_steps=max_steps)
    env.cube = cube  # Use our manually scrambled cube
    env.current_step = 0
    env.agent_moves = []
    env.scramble_sequence = scramble_moves  # Store the scramble sequence for consistency
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with the same architecture as in training
    state_size = 6 * 9 * 6  # 6 faces, 9 stickers per face, 6 possible colors
    action_size = len(MOVES)
    model = DQN(state_size, action_size).to(device)
    
    # Load the trained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    
    # Get initial state
    state = env._get_state()
    solution_moves = []
    solved = False
    steps = 0
    
    print("\nSolving...")
    while steps < max_steps:
        # Get action from model
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = torch.argmax(model(state_tensor)).item()
        
        # Apply action
        move = MOVES[action]
        solution_moves.append(move)
        print(f"Step {steps+1}: {move}")
        
        state, _, done = env.step(action)
        steps += 1
        
        # Check if solved
        if done and env.cube.is_solved():
            solved = True
            print(f"\nCube solved in {steps} moves!")
            break
        
        if done:
            print("\nFailed to solve within max steps.")
            break
    
    if not solved:
        print("\nFailed to solve the cube.")
    
    print(f"\nScramble: {scramble_sequence}")
    print(f"Solution attempt: {' '.join(solution_moves)}")
    
    return solved, solution_moves

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the Rubik\'s Cube RL agent')
    parser.add_argument('--model', type=str, default=None, 
                        help='Path to the model checkpoint (default: latest checkpoint)')
    parser.add_argument('--scramble', type=int, default=1, 
                        help='Number of scramble moves to test with (default: 1)')
    parser.add_argument('--tests', type=int, default=100, 
                        help='Number of test cases to run (default: 100)')
    parser.add_argument('--manual', type=str,
                        help='Manually specified scramble sequence (e.g., "R U F\' L2")')
    parser.add_argument('--use_pregenerated', action='store_true',
                        help='Use pregenerated scrambles from scrambles folder')
    
    args = parser.parse_args()
    failed = []
    try:
        if args.manual:
            solve_manual_scramble(args.manual, model_path=args.model)
        else:
            test_agent(model_path=args.model, scramble_moves=args.scramble, 
                      num_tests=args.tests, use_pregenerated=args.use_pregenerated)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using cube_rl.py.") 
import torch
import numpy as np
import random
import os
import time
import kociemba as koc
from cube import Cube
from cube_rl import CubeEnvironment, MOVES, DQN
from test_rl_agent import find_latest_checkpoint

class AdvancedCubeSolver:
    """
    Advanced solver that tries multiple strategies when the basic model fails.
    Implements a series of fallback mechanisms to improve solve rate.
    """
    
    def __init__(self, model_path=None, max_steps=50, max_retries=3, exploration_factor=0.2):
        """
        Initialize the advanced solver with multiple solving strategies.
        
        Args:
            model_path: Path to the model checkpoint (None = use latest)
            max_steps: Maximum steps per solving attempt
            max_retries: Maximum number of retry attempts with different strategies
            exploration_factor: Factor to control exploration in alternative strategies
        """
        # Find latest checkpoint if not specified
        if model_path is None:
            model_path = find_latest_checkpoint()
            if model_path is None:
                raise FileNotFoundError("No checkpoint files found. Please train the model first.")
            print(f"Using latest checkpoint: {model_path}")
        
        self.model_path = model_path
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.exploration_factor = exploration_factor
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model with the same architecture as in training
        state_size = 6 * 9 * 6  # 6 faces, 9 stickers per face, 6 possible colors
        action_size = len(MOVES)
        self.model = DQN(state_size, action_size).to(self.device)
        
        # Load the trained model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Set to evaluation mode
    
    def solve(self, scramble_sequence=None, scramble_cube=None, verbose=True):
        """
        Solve a cube using multiple strategies.
        
        Args:
            scramble_sequence: String of moves or list of moves (e.g., "R U F' L2" or ["R", "U", "F'", "L2"])
            scramble_cube: Already scrambled Cube object (alternative to scramble_sequence)
            verbose: Whether to print progress messages
            
        Returns:
            success (bool), solution_moves (list), strategy_used (str)
        """
        if verbose:
            print("=== Advanced Cube Solver ===")
        
        # Prepare the cube
        if scramble_cube is not None:
            # Use the provided cube
            cube = scramble_cube
            scramble_str = "Custom scrambled cube"
        elif scramble_sequence is not None:
            # Create a fresh cube and apply the scramble
            cube = Cube()
            if isinstance(scramble_sequence, str):
                scramble_moves = scramble_sequence.split()
            else:
                scramble_moves = scramble_sequence
                
            scramble_str = " ".join(scramble_moves)
            if verbose:
                print(f"Applying scramble: {scramble_str}")
                
            for move in scramble_moves:
                cube.apply_algorithm(move)
        else:
            raise ValueError("Either scramble_sequence or scramble_cube must be provided")
        
        # Try standard approach first
        if verbose:
            print("\n1. Trying standard approach...")
        success, solution_moves = self._standard_solve(cube.copy(), verbose)
        
        if success:
            if verbose:
                print(f"Cube solved using standard approach in {len(solution_moves)} moves.")
            return success, solution_moves, "standard"
        
        # Try different strategies
        strategies = [
            ("random restart", self._random_restart_solve),
            ("temperature exploration", self._temperature_explore_solve),
            ("breadth search", self._breadth_search_solve),
            # ("reverse moves", self._reverse_moves_solve)
        ]
        
        for strategy_name, strategy_fn in strategies:
            if verbose:
                print(f"\nStandard solve failed. Trying alternative approach: {strategy_name}...")
            
            success, solution_moves = strategy_fn(cube.copy(), verbose)
            
            if success:
                if verbose:
                    print(f"Cube solved using {strategy_name} approach in {len(solution_moves)} moves.")
                return success, solution_moves, strategy_name
        
        if verbose:
            print("\nAll solving approaches failed.")
        return False, [], "failed"
    
    def _standard_solve(self, cube, verbose=True):
        """Use the standard DQN model to solve the cube"""
        env = CubeEnvironment(max_steps=self.max_steps)
        env.cube = cube
        env.current_step = 0
        env.agent_moves = []
        
        state = env._get_state()
        solution_moves = []
        steps = 0
        
        while steps < self.max_steps:
            # Get action from model
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = torch.argmax(self.model(state_tensor)).item()
            
            # Apply action
            move = MOVES[action]
            solution_moves.append(move)
            if verbose and steps % 5 == 0:
                print(f"Step {steps+1}: {move}")
            
            state, _, done = env.step(action)
            steps += 1
            
            # Check if solved
            if done and env.cube.is_solved():
                return True, solution_moves
            
            if done:
                return False, solution_moves
        
        return False, solution_moves
    
    def _random_restart_solve(self, cube, verbose=True):
        """
        Try to solve by attempting each of the 18 possible moves as a starting point,
        then continue with the model.
        """
        best_solution = []
        best_distance = float('inf')
        
        # Try each of the 18 possible moves as a starting point
        for move_idx in range(len(MOVES)):
            if verbose:
                print(f"Trying move {move_idx+1}/18: {MOVES[move_idx]}")
            
            # Create a copy of the cube for this attempt
            test_cube = cube.copy()
            env = CubeEnvironment(max_steps=self.max_steps)
            env.cube = test_cube
            env.current_step = 0
            env.agent_moves = []
            
            state = env._get_state()
            solution_moves = []
            
            # Apply the current move
            move = MOVES[move_idx]
            solution_moves.append(move)
            state, _, done = env.step(move_idx)
            
            # If we somehow solve it with this move, great!
            if done and env.cube.is_solved():
                return True, solution_moves
            
            # Continue with model-guided actions
            steps = 1  # We've already made one move
            while steps < self.max_steps:
                # Get action from model
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = torch.argmax(self.model(state_tensor)).item()
                
                # Apply action
                move = MOVES[action]
                solution_moves.append(move)
                
                state, _, done = env.step(action)
                steps += 1
                
                # Check if solved
                if done and env.cube.is_solved():
                    return True, solution_moves
                
                if done:
                    break
            
            # If not solved, use kociemba to estimate how close we got
            try:
                solution_str = koc.solve(env.cube.to_kociemba_string())
                distance = len(solution_str.split())
                if distance < best_distance:
                    best_distance = distance
                    best_solution = solution_moves
                    if verbose:
                        print(f"  New best attempt: {len(solution_moves)} moves, distance: {distance}")
            except:
                # Invalid cube state, skip this attempt
                pass
        
        return False, best_solution
    
    def _temperature_explore_solve(self, cube, verbose=True):
        """Use temperature-based exploration to encourage more diverse moves"""
        env = CubeEnvironment(max_steps=self.max_steps)
        env.cube = cube
        env.current_step = 0
        env.agent_moves = []
        
        state = env._get_state()
        solution_moves = []
        steps = 0
        
        temperature = 1.0  # Starting temperature
        
        while steps < self.max_steps:
            # Get Q-values from model
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy()[0]
            
            # Apply temperature to encourage exploration
            if steps > 5:  # Use deterministic policy for first few steps
                q_values = q_values / temperature
            
            # Convert to probabilities using softmax
            exp_q = np.exp(q_values - np.max(q_values))
            probabilities = exp_q / np.sum(exp_q)
            
            # Sample action using these probabilities
            action = np.random.choice(len(MOVES), p=probabilities)
            
            # Apply action
            move = MOVES[action]
            solution_moves.append(move)
            
            if verbose and steps % 5 == 0:
                print(f"Step {steps+1}: {move} (temp: {temperature:.2f})")
            
            state, _, done = env.step(action)
            steps += 1
            
            # Check if solved
            if done and env.cube.is_solved():
                return True, solution_moves
            
            if done:
                return False, solution_moves
            
            # Reduce temperature slowly to focus more on best actions over time
            temperature = max(0.5, temperature * 0.98)
        
        return False, solution_moves
    
    def _breadth_search_solve(self, cube, verbose=True):
        """Try a limited breadth-first search approach to escape local minima"""
        # Limit this to only explore a small number of states to keep it tractable
        max_breadth = 3  # Maximum branching factor
        search_depth = 3  # How many moves to look ahead
        
        env = CubeEnvironment(max_steps=self.max_steps)
        env.cube = cube
        env.current_step = 0
        env.agent_moves = []
        
        state = env._get_state()
        solution_moves = []
        steps = 0
        
        while steps < self.max_steps:
            # If we're nearly out of steps, use the standard approach
            if steps > self.max_steps - search_depth:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = torch.argmax(self.model(state_tensor)).item()
                
                move = MOVES[action]
                solution_moves.append(move)
                state, _, done = env.step(action)
                steps += 1
                
                if done and env.cube.is_solved():
                    return True, solution_moves
                if done:
                    return False, solution_moves
                
                continue
            
            # Get Q-values for current state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy()[0]
            
            # Get the top N actions
            top_actions = np.argsort(-q_values)[:max_breadth]
            
            # Evaluate each action by looking ahead
            best_action = None
            best_value = float('-inf')
            
            for action in top_actions:
                # Create a temporary environment for simulation
                temp_env = CubeEnvironment(max_steps=self.max_steps)
                temp_env.cube = env.cube.copy()
                temp_env.current_step = env.current_step
                
                # Apply the action
                next_state, _, done = temp_env.step(action)
                
                if done and temp_env.cube.is_solved():
                    # Found a direct solution
                    best_action = action
                    break
                
                if done:
                    # Reached max steps, don't pick this action
                    continue
                
                # Simulate a few more moves with greedy policy
                cumulative_value = q_values[action]
                curr_state = next_state
                solved = False
                
                for d in range(search_depth - 1):
                    state_tensor = torch.FloatTensor(curr_state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        next_q_values = self.model(state_tensor).cpu().numpy()[0]
                    
                    next_action = np.argmax(next_q_values)
                    cumulative_value += next_q_values[next_action] * (0.95 ** (d+1))  # Discounted value
                    
                    next_state, _, done = temp_env.step(next_action)
                    curr_state = next_state
                    
                    if done and temp_env.cube.is_solved():
                        solved = True
                        break
                    
                    if done:
                        break
                
                # If this path led to a solution, prioritize it
                if solved:
                    best_action = action
                    break
                
                # Otherwise, pick the action with best cumulative value
                if cumulative_value > best_value:
                    best_value = cumulative_value
                    best_action = action
            
            # If no good action was found, fall back to greedy
            if best_action is None:
                best_action = np.argmax(q_values)
            
            # Apply the chosen action
            move = MOVES[best_action]
            solution_moves.append(move)
            
            if verbose and steps % 5 == 0:
                print(f"Step {steps+1}: {move} (breadth search)")
            
            state, _, done = env.step(best_action)
            steps += 1
            
            # Check if solved
            if done and env.cube.is_solved():
                return True, solution_moves
            
            if done:
                return False, solution_moves
        
        return False, solution_moves
    
    # def _reverse_moves_solve(self, cube, verbose=True):
    #     """
    #     If we detect we're going in circles, take a few steps back and try
    #     a different approach.
    #     """
    #     env = CubeEnvironment(max_steps=self.max_steps)
    #     env.cube = cube
    #     env.current_step = 0
    #     env.agent_moves = []
        
    #     state = env._get_state()
    #     solution_moves = []
    #     steps = 0
        
    #     # Keep track of visited states to detect loops
    #     visited_states = {}
        
    #     while steps < self.max_steps:
    #         # Convert state to a hashable form for tracking
    #         state_hash = self._hash_state(state)
            
    #         # Check if we've seen this state a lot
    #         if state_hash in visited_states:
    #             visited_states[state_hash] += 1
                
    #             # If we've revisited the same state multiple times, backtrack
    #             if visited_states[state_hash] >= 3:
    #                 if verbose:
    #                     print(f"Detected loop! Backtracking...")
                    
    #                 # Reverse the last few moves
    #                 back_steps = min(3, len(solution_moves))
                    
    #                 # Temporarily reset the cube to previous state by applying inverse moves
    #                 temp_cube = env.cube.copy()
    #                 back_moves = self._get_inverse_moves(solution_moves[-back_steps:])
    #                 for move in back_moves:
    #                     temp_cube.apply_algorithm(move)
                    
    #                 # Set up a new environment with the backtracked state
    #                 env.cube = temp_cube
    #                 env.current_step = max(0, env.current_step - back_steps)
                    
    #                 # Update solution_moves
    #                 solution_moves = solution_moves[:-back_steps]
                    
    #                 # Get the new state
    #                 state = env._get_state()
    #                 continue
    #         else:
    #             visited_states[state_hash] = 1
            
    #         # Get action from model, but exclude recently used moves if in a potential loop
    #         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #         with torch.no_grad():
    #             q_values = self.model(state_tensor).cpu().numpy()[0]
            
    #         # If we're possibly in a loop, penalize recently used actions
    #         if len(solution_moves) >= 4:
    #             recent_actions = [MOVES.index(move) for move in solution_moves[-4:]]
    #             for action in recent_actions:
    #                 q_values[action] *= 0.8  # Reduce preference for recent moves
            
    #         action = np.argmax(q_values)
            
    #         # Apply action
    #         move = MOVES[action]
    #         solution_moves.append(move)
            
    #         if verbose and steps % 5 == 0:
    #             print(f"Step {steps+1}: {move}")
            
    #         state, _, done = env.step(action)
    #         steps += 1
            
    #         # Check if solved
    #         if done and env.cube.is_solved():
    #             return True, solution_moves
            
    #         if done:
    #             return False, solution_moves
        
    #     return False, solution_moves
    
    # def _hash_state(self, state):
    #     """Convert state to a hashable form for detecting loops"""
    #     # Use the rounded values to create a string representation
    #     return hash(tuple(np.round(state, 2)))
    
    def _get_inverse_moves(self, moves):
        """Get the inverse sequence of moves to undo a sequence"""
        inverse_mapping = {
            "U": "U'", "U'": "U", "U2": "U2",
            "D": "D'", "D'": "D", "D2": "D2",
            "L": "L'", "L'": "L", "L2": "L2",
            "R": "R'", "R'": "R", "R2": "R2",
            "F": "F'", "F'": "F", "F2": "F2",
            "B": "B'", "B'": "B", "B2": "B2"
        }
        return [inverse_mapping[move] for move in reversed(moves)]

def solve_with_retry(scramble_sequence=None, scramble_cube=None, model_path=None,
                    max_steps=50, max_retries=3, verbose=True):
    """
    Convenience function to solve a cube with the advanced solver.
    
    Args:
        scramble_sequence: String of moves or list of moves
        scramble_cube: Already scrambled Cube object
        model_path: Path to model checkpoint
        max_steps: Maximum steps per solving attempt
        max_retries: Maximum number of retry attempts
        verbose: Whether to print detailed logs
        
    Returns:
        success (bool), solution_moves (list), strategy_used (str)
    """
    solver = AdvancedCubeSolver(
        model_path=model_path,
        max_steps=max_steps,
        max_retries=max_retries
    )
    
    return solver.solve(
        scramble_sequence=scramble_sequence,
        scramble_cube=scramble_cube,
        verbose=verbose
    )

def solve_benchmark(num_tests=100, scramble_moves=1, model_path=None, use_pregenerated=False):
    """
    Benchmark the advanced solving approach against the standard approach.
    
    Args:
        num_tests: Number of cubes to test
        scramble_moves: Number of moves to scramble the cube
        model_path: Path to model checkpoint
        use_pregenerated: Whether to use pregenerated scrambles
        
    Returns:
        A comparison of success rates between standard and advanced approaches
    """
    from test_rl_agent import test_agent, load_scrambles_from_file
    
    print(f"=== Benchmarking Advanced Solver vs Standard Solver ===")
    print(f"Testing {num_tests} cubes with {scramble_moves} move scrambles")
    
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
    
    # Initialize environment and solver
    env = CubeEnvironment(max_steps=50, scramble_moves=scramble_moves)
    solver = AdvancedCubeSolver(model_path=model_path, max_steps=50)
    
    # Statistics
    standard_solved = 0
    advanced_solved = 0
    standard_steps = 0
    advanced_steps = 0
    strategies_used = {}
    
    for i in range(num_tests):
        # Get a scrambled cube
        if use_pregenerated and i < len(pregenerated_scrambles):
            # Use a pregenerated scramble
            cube = Cube()
            scramble_data = pregenerated_scrambles[i]
            scramble = scramble_data["scramble"]
            
            # Apply the scramble
            cube.apply_algorithm(scramble)
            scramble_str = scramble
        else:
            # Use random scrambles
            cube = Cube()
            for _ in range(scramble_moves):
                move = random.choice(MOVES)
                cube.apply_algorithm(move)
            scramble_str = "Random scramble"
        
        # Test with standard approach
        standard_cube = cube.copy()
        env.cube = standard_cube
        env.current_step = 0
        env.agent_moves = []
        state = env._get_state()
        
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create and load model
        state_size = 6 * 9 * 6
        action_size = len(MOVES)
        model = DQN(state_size, action_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Solve with standard approach
        standard_solution = []
        standard_solved_this = False
        steps = 0
        
        while steps < 50:  # Max steps
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()
            
            standard_solution.append(MOVES[action])
            state, _, done = env.step(action)
            steps += 1
            
            if done and env.cube.is_solved():
                standard_solved += 1
                standard_solved_this = True
                standard_steps += steps
                break
            
            if done:
                break
        
        # Test with advanced approach (only if standard failed)
        if not standard_solved_this:
            # Create fresh copy of the original scrambled cube
            advanced_cube = cube.copy()
            success, solution_moves, strategy = solve_with_retry(
                scramble_cube=advanced_cube, 
                model_path=model_path,
                verbose=False
            )
            
            if success:
                advanced_solved += 1
                advanced_steps += len(solution_moves)
                strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
        
        # Print progress
        if (i+1) % 10 == 0 or i == 0 or i == num_tests - 1:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"=== Benchmark Progress: {i+1}/{num_tests} ===")
            print(f"Standard Solver: {standard_solved}/{i+1} ({standard_solved/(i+1)*100:.2f}%)")
            print(f"Advanced Solver: {standard_solved + advanced_solved}/{i+1} ({(standard_solved + advanced_solved)/(i+1)*100:.2f}%)")
            print(f"Improvement: +{advanced_solved} cubes (+{advanced_solved/(i+1)*100:.2f}%)")
            
            if strategies_used:
                print("\nStrategies Used:")
                for strategy, count in strategies_used.items():
                    print(f"- {strategy}: {count} successes")
    
    # Calculate final statistics
    standard_success_rate = (standard_solved / num_tests) * 100
    advanced_success_rate = ((standard_solved + advanced_solved) / num_tests) * 100
    improvement = advanced_success_rate - standard_success_rate
    
    standard_avg_steps = standard_steps / standard_solved if standard_solved > 0 else 0
    advanced_avg_steps = advanced_steps / advanced_solved if advanced_solved > 0 else 0
    
    # Calculate failure statistics
    standard_failed = num_tests - standard_solved
    advanced_failed = num_tests - (standard_solved + advanced_solved)
    recovered_from_failure = advanced_solved
    
    # Display results
    print("\n=== Final Benchmark Results ===")
    print(f"Scramble Moves: {scramble_moves}")
    print(f"Number of Tests: {num_tests}")
    print(f"Model: {model_path}")
    print()
    print(f"Standard Success Rate: {standard_success_rate:.2f}%")
    print(f"Advanced Success Rate: {advanced_success_rate:.2f}%")
    print(f"Improvement: +{improvement:.2f}%")
    print()
    print(f"Standard Avg Steps: {standard_avg_steps:.2f}")
    print(f"Advanced Avg Steps when Standard Failed: {advanced_avg_steps:.2f}")
    print()
    
    # New failure analysis section
    print("=== Failure Analysis ===")
    print(f"Standard approach failed: {standard_failed} cubes ({standard_failed/num_tests*100:.2f}%)")
    print(f"Advanced approach recovered: {recovered_from_failure} cubes ({recovered_from_failure/standard_failed*100:.2f}% of failures)")
    print(f"Remaining failures: {advanced_failed} cubes ({advanced_failed/num_tests*100:.2f}%)")
    print()
    
    if strategies_used:
        print("Successful strategies breakdown:")
        for strategy, count in strategies_used.items():
            percentage = (count / advanced_solved) * 100 if advanced_solved > 0 else 0
            print(f"- {strategy}: {count} cubes ({percentage:.2f}% of advanced solutions)")
    
    return standard_success_rate, advanced_success_rate

def solve_from_input():
    """
    A public function that prompts the user for a scramble and solves it.
    This can be called directly from other modules like cube.py.
    
    Example:
        from advanced_solver import solve_from_input
        solve_from_input()
    """
    print("\n=== Advanced Cube Solver Interface ===")
    
    # Get the scramble from the user
    scramble = input("Enter a scramble sequence (e.g., 'R U F' L2'): ")
    
    try:
        # Use the advanced solver with the scramble
        success, solution, strategy = solve_with_retry(
            scramble_sequence=scramble,
            verbose=True
        )
        
        # Display the results
        if success:
            print(f"\nSolution found using {strategy} strategy!")
            print(f"Solution: {' '.join(solution)}")
            print(f"Solution length: {len(solution)} moves")
            return success, solution, strategy
        else:
            print("\nFailed to solve the cube with all strategies.")
            return False, [], "failed"
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False, [], "error"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Rubik\'s Cube Solver with Multiple Strategies')
    parser.add_argument('--model', type=str, default=None, 
                        help='Path to the model checkpoint (default: latest checkpoint)')
    parser.add_argument('--scramble', type=str, default=None,
                        help='Manually specified scramble sequence (e.g., "R U F\' L2")')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark comparing standard vs advanced solver')
    parser.add_argument('--tests', type=int, default=100,
                        help='Number of test cases for benchmark (default: 100)')
    parser.add_argument('--scramble_moves', type=int, default=1,
                        help='Number of scramble moves for benchmark (default: 1)')
    parser.add_argument('--use_pregenerated', action='store_true',
                        help='Use pregenerated scrambles from scrambles folder')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode to solve user-provided scrambles')
    
    args = parser.parse_args()
    
    try:
        if args.benchmark:
            solve_benchmark(
                num_tests=args.tests, 
                scramble_moves=args.scramble_moves,
                model_path=args.model,
                use_pregenerated=args.use_pregenerated
            )
        elif args.scramble:
            success, solution, strategy = solve_with_retry(
                scramble_sequence=args.scramble,
                model_path=args.model
            )
            if success:
                print(f"\nSolution found using strategy: {strategy}")
                print(f"Solution: {' '.join(solution)}")
            else:
                print("\nFailed to solve the cube with all strategies.")
        elif args.interactive:
            solve_from_input()
        else:
            print("Please specify either --scramble, --benchmark, or --interactive.")
            print("Example: python advanced_solver.py --scramble \"R U F'\"")
            print("Example: python advanced_solver.py --benchmark --scramble_moves 3 --tests 50")
            print("Example: python advanced_solver.py --interactive")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using cube_rl.py.") 
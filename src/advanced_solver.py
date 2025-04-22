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
        
        # Define macro-operators for solving specific patterns
        self.macros = {
            "corner_swap": ["R", "U", "R'", "U'", "R'", "F", "R", "F'"],  # Sexy move
            "edge_flip": ["R", "U", "R'", "U", "R", "U2", "R'", "U"],     # Sune
            "corner_twist": ["R'", "D'", "R", "D"],                       # Corner orientation
            "center_rotation": ["F", "R", "U", "R'", "U'", "F'"]          # Sledgehammer
        }
        
        # For logging control
        self._last_step_logged = 0
    
    def _log_step(self, step, move, steps_total, verbose=True, label=None, always_show=False):
        """Helper method for cleaner logging during solving process"""
        if not verbose:
            return
            
        # Only log every few steps to keep output cleaner, unless always_show is True
        if not always_show and step % 5 != 0 and step != steps_total - 1:
            return
            
        # Update last logged step
        self._last_step_logged = step
        
        # Format the step string
        step_str = f"Step {step+1}/{steps_total}: {move}"
        if label:
            step_str += f" ({label})"
            
        print(step_str)
    
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
        
        # Strategy tiers ordered by computational complexity and effectiveness
        # Tier 1: Fast strategies (minimal computation, try these first)
        # Tier 2: Medium strategies (moderate computation)
        # Tier 3: Slow, complex strategies (expensive computation, try these last)
        
        strategy_tiers = [
            # Tier 1: Fast strategies
            [
                ("random restart", self._random_restart_solve, 5), 
                ("temperature exploration", self._temperature_explore_solve, 5), 
            ],
            # Tier 2: Medium strategies 
            [
                ("macro operators", self._macro_operators_solve, 5),  
                ("breadth search", self._breadth_search_solve, 5),  
            ],
            # Tier 3: Slow, complex strategies
            [
                ("monte carlo tree search", self._mcts_solve, 5),  
                ("beam search", self._beam_search_solve, 5),  
            ]
        ]
        
        # Attempt strategies in order of tiers
        for tier_idx, tier in enumerate(strategy_tiers):
            if verbose:
                print(f"\nTrying tier {tier_idx+1} strategies...")
            
            for strategy_idx, (strategy_name, strategy_fn, max_retries) in enumerate(tier):
                if verbose:
                    # Clear previous strategy output by printing over it
                    if tier_idx > 0 or strategy_idx > 0:
                        # Move cursor up 1 line and clear to end of screen
                        print("\033[1A\033[J", end="")
                    print(f"Tier {tier_idx+1}: Trying {strategy_name} approach...")
                
                # Try the strategy with multiple retries if configured
                for attempt in range(max_retries):
                    # Only show retry message if we're actually retrying
                    if attempt > 0 and verbose:
                        # Clear the previous attempt output
                        print("\033[1A\033[J", end="")
                        print(f"Tier {tier_idx+1}: {strategy_name} - Retry {attempt}/{max_retries-1}...")
                    
                    # Create modified parameters for retries to explore different parts of the search space
                    retry_params = {}
                    if strategy_name == "monte carlo tree search":
                        # Vary the exploration constant
                        retry_params["exploration_factor"] = 1.0 + (attempt * 0.3)
                    elif strategy_name == "beam search":
                        # Vary the beam width
                        retry_params["beam_width"] = 3 + attempt
                    elif strategy_name == "random restart":
                        # No special parameters needed, the strategy is already random
                        pass
                    
                    # Run the strategy with retry parameters
                    success, solution_moves = self._run_strategy_with_params(strategy_fn, cube.copy(), retry_params, verbose)
                    
                    if success:
                        if verbose:
                            print(f"Cube solved using {strategy_name} approach (attempt {attempt+1}) in {len(solution_moves)} moves.")
                        return success, solution_moves, f"{strategy_name} (retry {attempt+1})"
        
        if verbose:
            print("\nAll solving approaches failed.")
        return False, [], "failed"
    
    def _run_strategy_with_params(self, strategy_fn, cube, params, verbose):
        """Run a strategy with modified parameters"""
        # Save original parameters
        original_params = {}
        
        try:
            # Set temporary parameters
            for param_name, param_value in params.items():
                if hasattr(self, param_name):
                    original_params[param_name] = getattr(self, param_name)
                    setattr(self, param_name, param_value)
            
            # Run the strategy
            return strategy_fn(cube, verbose)
            
        finally:
            # Restore original parameters
            for param_name, param_value in original_params.items():
                setattr(self, param_name, param_value)
    
    def _standard_solve(self, cube, verbose=True):
        """Use the standard DQN model to solve the cube"""
        env = CubeEnvironment(max_steps=self.max_steps)
        env.cube = cube
        env.current_step = 0
        env.agent_moves = []
        
        state = env._get_state()
        solution_moves = []
        steps = 0
        
        self._last_step_logged = -1  # Reset logging counter
        
        while steps < self.max_steps:
            # Get action from model
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = torch.argmax(self.model(state_tensor)).item()
            
            # Apply action
            move = MOVES[action]
            solution_moves.append(move)
            
            # Log step (but keep it clean)
            self._log_step(steps, move, self.max_steps, verbose)
            
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
                # Clear previous attempt message if not the first one
                if move_idx > 0:
                    print("\033[1A\033[J", end="")
                print(f"Random restart: trying move {move_idx+1}/18: {MOVES[move_idx]}")
            
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
            self._last_step_logged = -1  # Reset logging counter
            
            while steps < self.max_steps:
                # Get action from model
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = torch.argmax(self.model(state_tensor)).item()
                
                # Apply action
                move = MOVES[action]
                solution_moves.append(move)
                
                # Log step with restart label
                restart_label = f"restart {move_idx+1}"
                self._log_step(steps, move, self.max_steps, verbose, restart_label)
                
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
        self._last_step_logged = -1  # Reset logging counter
        
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
            
            # Log step with temperature label
            temp_label = f"temp: {temperature:.2f}"
            self._log_step(steps, move, self.max_steps, verbose, temp_label)
            
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
        
        self._last_step_logged = -1  # Reset logging counter
        
        while steps < self.max_steps:
            # If we're nearly out of steps, use the standard approach
            if steps > self.max_steps - search_depth:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = torch.argmax(self.model(state_tensor)).item()
                
                move = MOVES[action]
                solution_moves.append(move)
                self._log_step(steps, move, self.max_steps, verbose, "standard")
                
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
            
            # Indicate we're doing breadth search analysis
            if verbose and steps % 5 == 0:
                print(f"Breadth search analysis at step {steps+1}...")
            
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
            
            # Log with breadth search label
            self._log_step(steps, move, self.max_steps, verbose, "breadth search")
            
            state, _, done = env.step(best_action)
            steps += 1
            
            # Check if solved
            if done and env.cube.is_solved():
                return True, solution_moves
            
            if done:
                return False, solution_moves
        
        return False, solution_moves
    
    def _mcts_solve(self, cube, verbose=True):
        """Monte Carlo Tree Search approach to find a solution"""
        # Maximum number of iterations for MCTS
        max_iterations = 200
        # Maximum search depth
        max_depth = min(20, self.max_steps)
        # Exploration constant for UCB1
        exploration_constant = 1.41
        
        class MCTSNode:
            def __init__(self, state, parent=None, action=None):
                self.state = state  # Cube state representation
                self.parent = parent  # Parent node
                self.action = action  # Action that led to this state
                self.children = {}  # Child nodes
                self.visits = 0  # Number of visits
                self.reward = 0  # Total reward
                self.untried_actions = list(range(len(MOVES)))  # Possible moves
                random.shuffle(self.untried_actions)  # Randomize action order
            
            def ucb1(self):
                """UCB1 formula for node selection"""
                if self.visits == 0:
                    return float('inf')
                return (self.reward / self.visits) + exploration_constant * np.sqrt(np.log(self.parent.visits) / self.visits)
            
            def select_child(self):
                """Select child with highest UCB1 value"""
                return max(self.children.values(), key=lambda node: node.ucb1())
            
            def expand(self, action, next_state):
                """Add a new child node"""
                child = MCTSNode(next_state, parent=self, action=action)
                self.untried_actions.remove(action)
                self.children[action] = child
                return child
            
            def update(self, reward):
                """Update node statistics"""
                self.visits += 1
                self.reward += reward
        
        # Setup environment
        env = CubeEnvironment(max_steps=self.max_steps)
        env.cube = cube
        state = env._get_state()
        
        # Create root node
        root = MCTSNode(state)
        
        # MCTS iterations
        iteration = 0
        best_solution = []
        best_reward = -float('inf')
        
        while iteration < max_iterations:
            if verbose and iteration % 20 == 0:
                # Clear previous iteration message if present
                if iteration > 0:
                    print("\033[1A\033[J", end="")
                print(f"MCTS iteration {iteration}/{max_iterations}")
            
            # Reset environment for this simulation
            sim_env = CubeEnvironment(max_steps=self.max_steps)
            sim_env.cube = cube.copy()
            sim_env.current_step = 0
            
            # Selection phase - navigate through tree until leaf node
            node = root
            path = []
            
            while not node.untried_actions and node.children:
                node = node.select_child()
                action = node.action
                path.append(MOVES[action])
                sim_env.step(action)
            
            # Check if current state is solved
            if sim_env.cube.is_solved():
                if verbose:
                    print(f"MCTS found solution during selection!")
                return True, path
            
            # Expansion phase - if node has untried actions, pick one randomly
            if node.untried_actions:
                action = node.untried_actions[0]  # Take first untried action
                next_state, _, done = sim_env.step(action)
                path.append(MOVES[action])
                
                # Check if solved after expansion
                if done and sim_env.cube.is_solved():
                    if verbose:
                        print(f"MCTS found solution during expansion!")
                    return True, path
                
                node = node.expand(action, next_state)
            
            # Simulation phase - random rollout to evaluate position
            rollout_depth = 0
            rollout_path = []
            
            while rollout_depth < max_depth:
                if sim_env.cube.is_solved():
                    break
                
                # Get action from model with some randomness
                state_tensor = torch.FloatTensor(sim_env._get_state()).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.model(state_tensor).cpu().numpy()[0]
                
                # Add randomness for exploration
                q_values = q_values + np.random.random(len(q_values)) * 0.1
                action = np.argmax(q_values)
                
                next_state, _, done = sim_env.step(action)
                rollout_path.append(MOVES[action])
                rollout_depth += 1
                
                if done:
                    break
            
            # Calculate reward
            if sim_env.cube.is_solved():
                reward = 10.0  # High reward for solved cube
                
                # Save solution if it's better than what we have
                solution = path + rollout_path
                if len(solution) < len(best_solution) or not best_solution:
                    best_solution = solution
                    if verbose:
                        # Clear previous solution message if present
                        print("\033[1A\033[J", end="")
                        print(f"MCTS found solution! Length: {len(best_solution)}")
                    return True, best_solution
            else:
                # Partial reward based on how close to solved
                try:
                    # Use Kociemba distance estimation as a heuristic
                    kociemba_str = sim_env.cube.to_kociemba_string()
                    solution_str = koc.solve(kociemba_str)
                    distance = len(solution_str.split())
                    reward = max(0, 20 - distance) / 20.0  # Normalize between 0 and 1
                except:
                    # If kociemba fails, use a simple heuristic
                    # Count correct colors on each face
                    correct_stickers = 0
                    for face in range(6):
                        center_color = sim_env.cube.state[face][4]  # Center sticker
                        for i in range(9):
                            if sim_env.cube.state[face][i] == center_color:
                                correct_stickers += 1
                    reward = correct_stickers / 54.0  # Normalize to [0,1]
            
            # Backpropagation - update statistics for all nodes in path
            while node:
                node.update(reward)
                node = node.parent
            
            iteration += 1
        
        # If we couldn't find a full solution, return the best partial solution
        if best_solution:
            return False, best_solution
        
        # If no good solutions found, use the most promising sequence from root
        if root.children:
            best_child = max(root.children.values(), key=lambda node: node.reward / max(1, node.visits))
            path = []
            
            # Reconstruct path from best child
            current = best_child
            while current.parent != root:
                path.insert(0, MOVES[current.action])
                current = current.parent
            
            path.insert(0, MOVES[best_child.action])
            return False, path
        
        return False, []
    
    def _macro_operators_solve(self, cube, verbose=True):
        """Use predefined macro-operators (move sequences) to solve common patterns"""
        env = CubeEnvironment(max_steps=self.max_steps)
        env.cube = cube
        env.current_step = 0
        env.agent_moves = []
        
        state = env._get_state()
        solution_moves = []
        steps = 0
        
        self._last_step_logged = -1  # Reset logging counter
        
        # We'll use a combination of model-based and macro-based approach
        while steps < self.max_steps:
            # Check if solved
            if env.cube.is_solved():
                return True, solution_moves
            
            # Every few steps, try to apply a macro if it improves the state
            if steps % 3 == 0:
                best_macro = None
                best_macro_score = float('-inf')
                best_macro_moves = []
                
                if verbose and steps % 5 == 0:
                    print(f"Evaluating macros at step {steps+1}...")
                
                # Try each macro and see which gives the best result
                for macro_name, macro_moves in self.macros.items():
                    # Create a copy of the current environment
                    test_env = CubeEnvironment(max_steps=self.max_steps)
                    test_env.cube = env.cube.copy()
                    test_env.current_step = env.current_step
                    
                    # Apply the macro
                    for move in macro_moves:
                        move_idx = MOVES.index(move)
                        test_env.step(move_idx)
                    
                    # Evaluate resulting state
                    try:
                        # Use kociemba distance as a heuristic
                        kociemba_str = test_env.cube.to_kociemba_string()
                        solution_str = koc.solve(kociemba_str)
                        distance = len(solution_str.split())
                        score = -distance  # Negative because shorter is better
                    except:
                        # Fallback heuristic: count correct stickers
                        correct_stickers = 0
                        for face in range(6):
                            center_color = test_env.cube.state[face][4]  # Center sticker
                            for i in range(9):
                                if test_env.cube.state[face][i] == center_color:
                                    correct_stickers += 1
                        score = correct_stickers
                    
                    # If this is the best macro so far, save it
                    if score > best_macro_score:
                        best_macro_score = score
                        best_macro = macro_name
                        best_macro_moves = macro_moves
                
                # If we found a useful macro, apply it
                if best_macro and best_macro_score > 0:
                    if verbose:
                        print(f"Applying macro '{best_macro}'")
                    
                    for move in best_macro_moves:
                        move_idx = MOVES.index(move)
                        solution_moves.append(move)
                        
                        # Log step with macro label
                        self._log_step(steps, move, self.max_steps, verbose, f"macro: {best_macro}", always_show=True)
                        
                        state, _, done = env.step(move_idx)
                        steps += 1
                        
                        if done and env.cube.is_solved():
                            return True, solution_moves
                        
                        if done or steps >= self.max_steps:
                            return False, solution_moves
                    
                    continue
            
            # If no macro was applied, use the model
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = torch.argmax(self.model(state_tensor)).item()
            
            # Apply the model-selected move
            move = MOVES[action]
            solution_moves.append(move)
            
            # Log step
            self._log_step(steps, move, self.max_steps, verbose)
            
            state, _, done = env.step(action)
            steps += 1
            
            # Check if solved
            if done and env.cube.is_solved():
                return True, solution_moves
            
            if done:
                return False, solution_moves
        
        return False, solution_moves
    
    def _beam_search_solve(self, cube, verbose=True):
        """Use beam search to explore multiple solution paths simultaneously"""
        # Beam width - number of states to explore at each level
        beam_width = 5
        # Maximum search depth
        max_depth = self.max_steps
        
        # Define a state node for beam search
        class BeamNode:
            def __init__(self, cube, moves=None, score=0):
                self.cube = cube
                self.moves = moves or []
                self.score = score
            
            def __lt__(self, other):
                return self.score > other.score  # For priority queue (higher score is better)
        
        # Function to evaluate a cube state
        def evaluate_cube(cube_state):
            try:
                # Use kociemba distance as a primary heuristic
                kociemba_str = cube_state.to_kociemba_string()
                solution_str = koc.solve(kociemba_str)
                distance = len(solution_str.split())
                score = 100 - 2 * distance  # Penalize longer solutions
            except:
                # Fallback heuristic: count correct stickers
                correct_stickers = 0
                for face in range(6):
                    center_color = cube_state.state[face][4]  # Center sticker
                    for i in range(9):
                        if cube_state.state[face][i] == center_color:
                            correct_stickers += 1
                score = correct_stickers
            
            return score
        
        # Initialize beam with the starting state
        beam = [BeamNode(cube.copy(), [], evaluate_cube(cube))]
        
        # Beam search
        depth = 0
        while depth < max_depth and beam:
            if verbose and depth % 5 == 0:
                # Clear previous depth message
                if depth > 0:
                    print("\033[1A\033[J", end="")
                print(f"Beam search depth: {depth}/{max_depth} (width: {beam_width})")
            
            # Generate all possible next states for all nodes in the beam
            next_beam = []
            
            for node in beam:
                # Check if this node is already solved
                if node.cube.is_solved():
                    if verbose:
                        print(f"Solution found at depth {depth}")
                    return True, node.moves
                
                # Get model predictions for this state
                env = CubeEnvironment()
                env.cube = node.cube
                state = env._get_state()
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.model(state_tensor).cpu().numpy()[0]
                
                # Get top actions from the model
                top_actions = np.argsort(-q_values)
                
                # Explore all possible moves for this node
                for action in top_actions[:beam_width]:  # Limit exploration to top actions
                    # Create a copy of the cube
                    new_cube = node.cube.copy()
                    
                    # Apply the move
                    move = MOVES[action]
                    new_cube.apply_algorithm(move)
                    
                    # Create new node
                    new_moves = node.moves + [move]
                    new_score = evaluate_cube(new_cube)
                    
                    # Add bonus for moves recommended by the model
                    model_confidence = q_values[action]
                    new_score += model_confidence * 0.5
                    
                    # Add to candidates
                    next_beam.append(BeamNode(new_cube, new_moves, new_score))
            
            # Select the best nodes for the next beam
            next_beam.sort()  # Sort by score (higher is better)
            beam = next_beam[:beam_width]  # Keep only the best nodes
            
            # Check if the best node is solved
            if beam and beam[0].cube.is_solved():
                if verbose:
                    print(f"Solution found at depth {depth+1}")
                return True, beam[0].moves
            
            depth += 1
        
        # If we reach here, we couldn't find a solution
        # Return the best partial solution
        if beam:
            best_node = beam[0]
            return False, best_node.moves
        
        return False, []
    
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

def solve_benchmark(num_tests=100, scramble_moves=1, model_path=None, use_pregenerated=False, max_steps=50):
    """
    Benchmark the advanced solving approach against the standard approach.
    
    Args:
        num_tests: Number of cubes to test
        scramble_moves: Number of moves to scramble the cube
        model_path: Path to model checkpoint
        use_pregenerated: Whether to use pregenerated scrambles
        max_steps: Maximum steps for solving attempts
        
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
    scrambles = []
    if use_pregenerated:
        scrambles = load_scrambles_from_file(scramble_moves, num_tests)
        if not scrambles:
            print(f"Could not load pregenerated scrambles for {scramble_moves} moves. Using random scrambles.")
            use_pregenerated = False
    
    # Generate random scrambles if needed
    if not use_pregenerated:
        for _ in range(num_tests):
            cube = Cube()
            scramble = []
            for _ in range(scramble_moves):
                move = random.choice(MOVES)
                scramble.append(move)
            scrambles.append({"scramble": scramble})
    
    # Ensure we have the correct number of scrambles
    scrambles = scrambles[:num_tests]
    
    # Initialize environment and prepare for standard solving
    env = CubeEnvironment(max_steps=max_steps, scramble_moves=scramble_moves)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create and load model
    state_size = 6 * 9 * 6
    action_size = len(MOVES)
    model = DQN(state_size, action_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Statistics
    standard_solved = 0
    advanced_solved = 0
    standard_steps = 0
    advanced_steps = 0
    strategies_used = {}
    
    # Store failed scrambles for later processing
    failed_scrambles = []
    failed_cubes = []
    
    # First phase: Try standard solver on all scrambles
    print("\n=== Phase 1: Testing Standard Solver ===")
    for i, scramble_data in enumerate(scrambles):
        # Get the scramble
        scramble = scramble_data["scramble"]
        
        # Create a fresh cube and apply the scramble
        cube = Cube()
        if isinstance(scramble, str):
            scramble_moves = scramble.split()
        else:
            scramble_moves = scramble
        
        for move in scramble_moves:
            cube.apply_algorithm(move)
        
        scramble_str = " ".join(scramble_moves) if isinstance(scramble_moves, list) else scramble_moves
        
        # Solve with standard approach
        env.cube = cube.copy()
        env.current_step = 0
        env.agent_moves = []
        state = env._get_state()
        
        standard_solution = []
        steps = 0
        
        while steps < max_steps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()
            
            standard_solution.append(MOVES[action])
            state, _, done = env.step(action)
            steps += 1
            
            if done and env.cube.is_solved():
                standard_solved += 1
                standard_steps += steps
                break
            
            if done:
                failed_scrambles.append(scramble_str)
                failed_cubes.append(cube.copy())
        
        # Update progress
        if (i+1) % 10 == 0 or i == 0 or i == num_tests - 1:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"=== Phase 1: Testing Standard Solver ===")
            print(f"Progress: {i+1}/{num_tests} cubes tested")
            print(f"Standard Solver Success: {standard_solved}/{i+1} ({standard_solved/(i+1)*100:.2f}%)")
            if standard_solved > 0:
                print(f"Average Standard Steps: {standard_steps/standard_solved:.2f}")
    
    # Final stats for phase 1
    standard_success_rate = (standard_solved / num_tests) * 100
    
    # Initialize the advanced solver
    solver = AdvancedCubeSolver(model_path=model_path, max_steps=max_steps)
    
    # Second phase: Try advanced strategies on failed scrambles
    print("\n=== Phase 2: Testing Advanced Strategies on Failed Scrambles ===")
    total_failures = len(failed_scrambles)
    
    if total_failures == 0:
        print("No failures to recover! Standard solver solved all cubes.")
    else:
        print(f"Trying to recover {total_failures} failed scrambles with advanced strategies...")
        
        for i, (scramble_str, cube) in enumerate(zip(failed_scrambles, failed_cubes)):
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Show overall progress stats at the top
            advanced_success_rate = ((standard_solved + advanced_solved) / num_tests) * 100
            improvement = advanced_success_rate - standard_success_rate
            
            print(f"=== Advanced Solver Progress ===")
            print(f"Standard Solver: {standard_solved}/{num_tests} ({standard_success_rate:.2f}%)")
            print(f"Advanced Solver: {standard_solved + advanced_solved}/{num_tests} ({advanced_success_rate:.2f}%)")
            print(f"Improvement: +{improvement:.2f}%")
            print(f"Failed scrambles remaining: {total_failures - i}/{total_failures}")
            
            if strategies_used:
                print("\nStrategies Used:")
                for strategy, count in strategies_used.items():
                    print(f"- {strategy}: {count} successes")
            
            print("\n=== Current Failed Scramble ===")
            print(f"Scramble ({i+1}/{total_failures}): {scramble_str}")
            
            # Get kociemba solution length if possible
            try:
                kociemba_solution = koc.solve(cube.to_kociemba_string())
                kociemba_length = len(kociemba_solution.split())
                print(f"Optimal solution length (kociemba): {kociemba_length} moves")
            except Exception:
                print("Unable to determine optimal solution length")
                
            print("-" * 50)  # Divider line
            
            # Define a clean output handler class for the solving process
            class SolveOutputManager:
                def __init__(self):
                    # Store a reference to the original print function
                    import builtins
                    self.original_print = builtins.print
                    self.strategy_displayed = False
                    
                def show_strategy(self, tier, strategy, retry=None):
                    # Clear previous strategy line if there was one
                    if self.strategy_displayed:
                        # Move cursor up 1 line and clear to end of line
                        self.original_print("\033[1A\033[K", end="")
                    
                    # Show current strategy info
                    if retry is None:
                        self.original_print(f"Strategy: Tier {tier} - {strategy}")
                    else:
                        self.original_print(f"Strategy: Tier {tier} - {strategy} (Retry {retry})")
                    
                    self.strategy_displayed = True
                
                def clear_solver_output(self):
                    # Clear multiple lines of solver output by using a clear screen below the header
                    self.original_print("\033[J", end="")
                    
                def print_success(self, message):
                    self.clear_solver_output()
                    self.original_print(f"✓ {message}")
                    
                def print_failure(self, message):
                    self.clear_solver_output()
                    self.original_print(f"✗ {message}")
            
            # Create output manager
            output_manager = SolveOutputManager()
            
            # Create a solver with the basic model
            solver = AdvancedCubeSolver(model_path=model_path, max_steps=max_steps)
            
            # Dictionary to map strategy functions to their names and tier
            strategy_info = {
                "_standard_solve": ("Standard", 0),
                "_random_restart_solve": ("Random Restart", 1),
                "_temperature_explore_solve": ("Temperature Exploration", 1),
                "_macro_operators_solve": ("Macro Operators", 2),
                "_breadth_search_solve": ("Breadth Search", 2),
                "_mcts_solve": ("Monte Carlo Tree Search", 3),
                "_beam_search_solve": ("Beam Search", 3)
            }
            
            # Try standard approach first
            output_manager.show_strategy(0, "Standard")
            success, solution_moves = solver._standard_solve(cube.copy(), verbose=False)
            
            if success:
                output_manager.print_success(f"Solved with Standard approach in {len(solution_moves)} moves")
                advanced_solved += 1
                advanced_steps += len(solution_moves)
                strategies_used["standard"] = strategies_used.get("standard", 0) + 1
            else:
                # Try all other strategies in order of tiers
                strategy_tiers = [
                    # Tier 1: Fast strategies
                    [
                        ("_random_restart_solve", 5), 
                        ("_temperature_explore_solve", 5), 
                    ],
                    # Tier 2: Medium strategies 
                    [
                        ("_macro_operators_solve", 5),  
                        ("_breadth_search_solve", 5),  
                    ],
                    # Tier 3: Slow, complex strategies
                    [
                        ("_mcts_solve", 5),  
                        ("_beam_search_solve", 5),  
                    ]
                ]
                
                solved = False
                
                # Loop through each tier
                for tier_idx, tier in enumerate(strategy_tiers):
                    if solved:
                        break
                    
                    # Loop through each strategy in the tier
                    for strategy_name, max_retries in tier:
                        if solved:
                            break
                        
                        # Get the display name and tier from our mapping
                        display_name = strategy_info[strategy_name][0]
                        tier_num = tier_idx + 1  # Tier is 1-indexed for display
                        
                        # Show initial strategy
                        output_manager.show_strategy(tier_num, display_name)
                        
                        # Get the strategy function
                        strategy_fn = getattr(solver, strategy_name)
                        
                        # Try the strategy
                        success, solution_moves = strategy_fn(cube.copy(), verbose=False)
                        
                        if success:
                            solved = True
                            output_manager.print_success(f"Solved with {display_name} approach in {len(solution_moves)} moves")
                            advanced_solved += 1
                            advanced_steps += len(solution_moves)
                            strategies_used[display_name] = strategies_used.get(display_name, 0) + 1
                            break
                        
                        # Try retries if needed
                        for retry in range(1, max_retries):
                            # Show retry
                            output_manager.show_strategy(tier_num, display_name, retry)
                            
                            # Create retry params
                            retry_params = {}
                            if strategy_name == "_mcts_solve":
                                retry_params["exploration_factor"] = 1.0 + (retry * 0.3)
                            elif strategy_name == "_beam_search_solve":
                                retry_params["beam_width"] = 3 + retry
                            
                            # Try with these parameters
                            temp_env = CubeEnvironment(max_steps=max_steps)
                            temp_env.cube = cube.copy()
                            
                            # Run the strategy - use the same solver but with retry params
                            success, solution_moves = solver._run_strategy_with_params(
                                strategy_fn, 
                                cube.copy(), 
                                retry_params, 
                                verbose=False
                            )
                            
                            if success:
                                solved = True
                                output_manager.print_success(f"Solved with {display_name} (Retry {retry}) in {len(solution_moves)} moves")
                                advanced_solved += 1
                                advanced_steps += len(solution_moves)
                                strategy_key = f"{display_name} (retry {retry})"
                                strategies_used[strategy_key] = strategies_used.get(strategy_key, 0) + 1
                                break
                
                # If we tried everything and still couldn't solve it
                if not solved:
                    output_manager.print_failure("Failed to solve with any strategy")
            
            # Pause briefly to see the result before moving on
            time.sleep(0.5)
    
    # Calculate final statistics
    os.system('cls' if os.name == 'nt' else 'clear')
    
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
    
    # Failure analysis section
    print("=== Failure Analysis ===")
    print(f"Standard approach failed: {standard_failed} cubes ({standard_failed/num_tests*100:.2f}%)")
    if standard_failed > 0:
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
    parser.add_argument('--max_steps', type=int, default=50,
                        help='Maximum steps for solving attempts (default: 50)')
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
                use_pregenerated=args.use_pregenerated,
                max_steps=args.max_steps
            )
        elif args.scramble:
            success, solution, strategy = solve_with_retry(
                scramble_sequence=args.scramble,
                model_path=args.model,
                max_steps=args.max_steps
            )
            if success:
                print(f"\nSolution found using strategy: {strategy}")
                print(f"Solution: {' '.join(solution)}")
            else:
                print("\nFailed to solve the cube with all strategies.")
        elif args.interactive:
            solve_from_input()
        else:
            # Default to benchmark with the user's requested parameters
            solve_benchmark(
                num_tests=10000,  # t=10000
                scramble_moves=7,  # n=7
                max_steps=6,      # m=6
                model_path=args.model,
                use_pregenerated=args.use_pregenerated
            )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using cube_rl.py.") 
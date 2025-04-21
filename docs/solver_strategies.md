# Rubik's Cube Solving Strategies
# Detailed explanation of all strategies used in the advanced solver

## Overview
The advanced solver implements multiple strategies in tiers of increasing complexity. Each strategy is designed to handle different types of cube states and has its own strengths and weaknesses.

## Strategy Tiers

### Tier 1: Fast Strategies
These strategies are computationally efficient and are tried first.

#### 1. Standard Approach
- Uses the trained DQN model directly
- Makes decisions based on the model's Q-values
- Fastest approach but may fail on complex positions
- No exploration or backtracking
- Best for simple scrambles and positions the model has seen during training

#### 2. Random Restart
- Tries each of the 18 possible moves as a starting point
- For each starting move, continues with the standard model
- Helps when the initial position is difficult for the model
- Good for positions where the first move is critical
- Limited by the model's ability to continue from each starting point

#### 3. Temperature Exploration
- Adds controlled randomness to the model's decisions
- Uses temperature parameter to control exploration
- Higher temperature = more exploration
- Lower temperature = more exploitation
- Good for positions where the model is unsure

### Tier 2: Medium Strategies
These strategies use more computation but can handle more complex positions.

#### 1. Macro Operators
- Uses predefined sequences of moves (macros) for specific patterns
- Examples:
  * Corner swap: ["R", "U", "R'", "U'", "R'", "F", "R", "F'"] (Sexy move)
  * Edge flip: ["R", "U", "R'", "U", "R", "U2", "R'", "U"] (Sune)
  * Corner twist: ["R'", "D'", "R", "D"]
  * Center rotation: ["F", "R", "U", "R'", "U'", "F'"] (Sledgehammer)
- Good for recognizing and solving common patterns
- Can solve specific sub-problems efficiently

#### 2. Breadth Search
- Explores multiple solution paths simultaneously
- Maintains a queue of promising positions
- Evaluates positions using multiple heuristics
- Good for finding shortest solutions
- Can handle positions with multiple solution paths

### Tier 3: Slow, Complex Strategies
These strategies are computationally expensive but can solve very difficult positions.

#### 1. Monte Carlo Tree Search (MCTS)
- Builds a tree of possible moves and their outcomes
- Four phases:
  1. Selection: Choose promising nodes using UCB1 formula
  2. Expansion: Try new moves from leaf nodes
  3. Simulation: Random playout to evaluate position
  4. Backpropagation: Update node statistics
- Uses multiple heuristics:
  * Kociemba distance estimation
  * Correct sticker count
  * Model Q-values
- Parameters:
  * Max iterations: 200
  * Max depth: 20 moves
  * Exploration constant: 1.41
- Excellent for:
  * Complex positions
  * Finding optimal solutions
  * Balancing exploration and exploitation

#### 2. Beam Search
- Maintains a beam of most promising positions
- Evaluates positions using multiple criteria:
  * Model Q-values
  * Distance to solved state
  * Move sequence length
- Good for:
  * Finding good solutions quickly
  * Handling positions with many possible moves
  * Balancing solution quality and computation time

## Strategy Selection and Fallback
The solver tries strategies in order of increasing complexity:
1. Starts with Tier 1 strategies (fast)
2. If unsuccessful, moves to Tier 2 (medium)
3. Finally tries Tier 3 (slow) if needed

Each strategy has multiple retry attempts with different parameters:
- MCTS: Varies exploration factor
- Beam Search: Varies beam width
- Random Restart: Naturally random
- Temperature Exploration: Varies temperature

## Evaluation Metrics
Strategies are evaluated using:
1. Success rate
2. Solution length
3. Computation time
4. Memory usage

## Best Practices
1. Start with faster strategies for simple positions
2. Use MCTS for complex positions
3. Combine strategies for optimal results
4. Adjust parameters based on position difficulty
5. Use pregenerated scrambles for consistent testing

## Performance Considerations
- Memory usage increases with search depth
- Computation time varies significantly between strategies
- Some strategies may need parameter tuning for specific positions
- Consider using parallel processing for expensive strategies 
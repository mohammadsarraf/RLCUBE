# Action Plan for Improving 5-Move Scramble Performance

Based on your challenges with 5-move scrambles, I recommend a three-phase approach:

## Phase 1: Enhanced Training (2-3 weeks)

### Curriculum Learning Implementation
- Start with 1-move scrambles until 99%+ success
- Progress to 2-move scrambles until 95%+ success
- Continue adding complexity gradually
- Use validation sets to prevent overfitting
- 100% for 1-2 moves
- 95%+ for 3-4 moves
- 80%+ for 5-6 moves
- 60%+ for 7-10 moves
- 40%+ for 11-15 moves
- 20%+ for 16-20 moves

### Training Data Improvements
- Generate 10,000+ diverse 5-move scrambles
- Ensure scrambles cover all move types and patterns
- Create "hard case" datasets focused on problematic patterns

### Model Architecture Enhancements
- Increase model capacity (more layers/neurons)
- Add residual connections to improve gradient flow
- Implement attention mechanisms to focus on important cube patterns
- Try different activation functions (Mish, GELU)

## Phase 2: Advanced Solver Optimization (1-2 weeks)

### Extend MCTS Parameters
- Increase max_iterations from 200 to 500+ for difficult scrambles
- Implement progressive widening for deep states
- Add pattern recognition to guide the search

### Create Specialized Hard-Case Strategies
- Develop targeted macro operators for 5-move patterns
- Implement pattern recognition for common 5-move scrambles
- Add decomposition strategy (solve subcomponents first)

### Dynamic Strategy Selection
- Create a predictor that chooses the best strategy based on cube state
- Allocate more computation to harder scrambles
- Implement strategy chaining (apply strategies in sequence)

## Phase 3: Evaluation and Refinement (1 week)

### Comprehensive Benchmarking
- Create benchmark suite with 1-10 move scrambles
- Analyze failure patterns
- Track success rates across different scramble types

### Hybrid Approach Fine-Tuning
- Balance RL model usage with search algorithms
- Optimize strategy selection based on benchmark data
- Identify specific patterns where strategies fail

## Recommended First Steps (This Week)

### Immediate Actions:
- Run extended training (2x current epochs) on 5-move scrambles
- Increase MCTS iterations to 500 for 5-move scrambles
- Implement a specialized 5-move pattern detector

### Data Collection:
- Generate 1,000 diverse 5-move scrambles
- Identify the most challenging patterns
- Create a validation set for measuring progress

### Quick Wins:
- Double the retries for MCTS and beam search on 5-move scrambles
- Implement a "last-resort" mode with extended computation for failures
- Add a pattern database for common 5-move positions

## Specific Implementation Recommendations

### 1. Curriculum Learning Optimization
- Lower target_success_rate for higher levels:
  - Levels 1-2: 95%+
  - Level 3: 70%
  - Level 4: 60%
  - Level 5: 50%
- Increase max_episodes:
  - Level 4: 25,000 episodes
  - Level 5: 50,000 episodes
- Add min_rate parameter: Force training to continue until at least 40% success rate

### 2. Training Data Generation
- Use existing scramble generation (gen.py) with modifications:
  - Generate 10,000+ diverse scrambles per level
  - Include "hard case" scrambles from failure analysis
  - Create specialized scramble sets for consistently failing patterns

### 3. Model Hyperparameter Tuning
Run experiments with these settings without changing code structure:
- Lower learning_rate: 0.0005 (instead of 0.001)
- Slower epsilon decay: 0.999 (instead of 0.998)
- Higher gamma: 0.98 or 0.99 (instead of 0.95)
- Larger batch_size: 128 or 256 (instead of 64)

### Command Line Arguments for Training
```bash
python cube_rl.py --level 1 --max_level 5 --min_episodes 5000 --max_episodes 50000 --target_rate 70 --min_rate 40 --batch_size 128 --use_pregenerated
```

### Expected Outcomes
- Standard solver: 60-70% success rate for 5-move scrambles
- Advanced solver: 98-100% success rate
- Leverage existing architecture while optimizing training process and data quality

python cube_rl.py --level 3 --max_level 3 --min_rate 92 --batch_size 128 --use_pregenerated

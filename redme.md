# Continue training from where you left off
python cube_rl.py

# Start training at a specific level
python cube_rl.py --level 2

# Train with custom parameters
python cube_rl.py --level 2 --max_level 4 --min_episodes 10000 --target_rate 40

# Train just one specific level
python cube_rl.py --level 3 --max_level 3

# Train with pregenerated scrambles
python cube_rl.py --use_pregenerated

# Train with custom batch size
python cube_rl.py --batch_size 128

# Full custom training example
python cube_rl.py --level 2 --max_level 5 --min_episodes 10000 --max_episodes 50000 --target_rate 30 --batch_size 64 --use_pregenerated

# Test with default settings (100 tests, 1 move scramble)
python test_rl_agent.py

# Test with specific number of scramble moves
python test_rl_agent.py --scramble 3

# Test with custom number of tests
python test_rl_agent.py --scramble 2 --tests 50

# Test with a specific scramble sequence
python test_rl_agent.py --manual "R U' F L2"

# Test with pregenerated scrambles
python test_rl_agent.py --use_pregenerated

# Run benchmark comparing standard vs advanced solver
python advanced_solver.py --benchmark

# Benchmark with custom parameters
python advanced_solver.py --benchmark --scramble_moves 3 --tests 50

# Benchmark with specific model
python advanced_solver.py --benchmark --model cube_solver_model_scramble_3.pt

# Solve a specific scramble
python advanced_solver.py --scramble "R U F' L2"

# Solve with specific model
python advanced_solver.py --scramble "R U F' L2" --model cube_solver_model_scramble_3.pt

# Generate scrambles for a specific level
python gen.py --level 3

# Generate custom number of scrambles
python gen.py --level 3 --count 10000
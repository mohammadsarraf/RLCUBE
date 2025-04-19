#!/bin/bash

# Scenario 1: Start training from scratch at level 1
echo "Scenario 1: Starting training from scratch at level 1..."
python cube_rl.py --level 1 --max_level 1 --min_rate 95 --use_pregenerated --target_rate 100

# Scenario 2: Continue training from the last checkpoint
echo "Scenario 2: Continuing training for level 2 using checkpoint from level 1..."
python cube_rl.py --level 2 --max_level 2 --min_rate 90 --use_pregenerated --target_rate 100

# Scenario 3: Improve an existing level using a checkpoint from a previous level
echo "Scenario 3: Improving level 1 using a checkpoint from level 1 (better training)..."
python cube_rl.py --level 1 --max_level 1 --min_rate 99 --use_pregenerated --target_rate 100 --model cube_solver_model_scramble_1.pt

python cube_rl.py --level 2 --max_level 2 --min_rate 98 --use_pregenerated --target_rate 100 --model cube_solver_model_scramble_2.pt

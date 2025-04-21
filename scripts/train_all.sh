#!/bin/bash
# This script has been updated to use parallel training

# Get the number of available CPU cores
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Detected $NUM_CORES CPU cores"

# Use 75% of available cores for training to avoid overwhelming the system
NUM_PROCESSES=$(($NUM_CORES * 3 / 4))
if [ $NUM_PROCESSES -lt 1 ]; then
    NUM_PROCESSES=1
fi
echo "Using $NUM_PROCESSES processes for parallel training"

# Option 1: Train all levels in one go (most parallel)
python parallel_cube_rl.py --mode train --level 5 --max_level 11 --min_rate 90 --use_pregenerated --target_rate 100 --min_episodes 50000 --batch_size 128 --recent_window 10000 --processes $NUM_PROCESSES

# Option 2: Train in batches (comment out Option 1 and uncomment below if you prefer this approach)
# echo "Training levels 5-7..."
# python parallel_cube_rl.py --mode train --level 5 --max_level 7 --min_rate 90 --use_pregenerated --target_rate 100 --min_episodes 50000 --batch_size 128 --recent_window 10000 --processes $NUM_PROCESSES
# 
# echo "Training levels 8-9..."
# python parallel_cube_rl.py --mode train --level 8 --max_level 9 --min_rate 90 --use_pregenerated --target_rate 100 --min_episodes 50000 --batch_size 128 --recent_window 10000 --processes $NUM_PROCESSES
# 
# echo "Training levels 10-11..."
# python parallel_cube_rl.py --mode train --level 10 --max_level 11 --min_rate 90 --use_pregenerated --target_rate 100 --min_episodes 50000 --batch_size 128 --recent_window 10000 --processes $NUM_PROCESSES

echo "All training completed!"

#!/bin/bash

echo "Starting training for levels 1-2 with 99% minimum success rate..."
python cube_rl.py --level 1 --max_level 2 --min_rate 99

echo "Starting training for levels 3-4 with 95% minimum success rate..."
python cube_rl.py --level 3 --max_level 4 --min_rate 95

echo "Starting training for level 5 with 80% minimum success rate..."
python cube_rl.py --level 5 --max_level 5 --min_rate 80

echo "Training completed for all levels!" 
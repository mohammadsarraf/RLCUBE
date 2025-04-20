#!/bin/bash
# Script to improve all existing checkpoints in parallel

# Get the number of available CPU cores
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Detected $NUM_CORES CPU cores"

# Use 75% of available cores for training to avoid overwhelming the system
NUM_PROCESSES=$(($NUM_CORES * 3 / 4))
if [ $NUM_PROCESSES -lt 1 ]; then
    NUM_PROCESSES=1
fi
echo "Using $NUM_PROCESSES processes for parallel improvement"

# Find all existing checkpoint levels
CHECKPOINT_DIR="modelCheckpoints"
LEVELS=""
for FILE in $CHECKPOINT_DIR/cube_solver_model_scramble_*.pt; do
    if [ -f "$FILE" ]; then
        LEVEL=$(echo $FILE | sed -E 's/.*_([0-9]+)\.pt/\1/')
        if [ -n "$LEVELS" ]; then
            LEVELS="$LEVELS,$LEVEL"
        else
            LEVELS="$LEVEL"
        fi
    fi
done

if [ -z "$LEVELS" ]; then
    echo "No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Found checkpoints for levels: $LEVELS"
echo "Starting parallel improvement..."

# Run parallel improvement for all levels
python parallel_cube_rl.py --mode improve --levels $LEVELS --min_rate 95 --target_rate 98 --use_pregenerated --batch_size 128 --min_episodes 10000 --max_episodes 20000 --processes $NUM_PROCESSES

echo "All improvements completed!" 
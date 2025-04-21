# Parallel Training for Rubik's Cube RL Agent

This README explains how to use the parallel training capabilities for faster training of the Rubik's Cube RL agent.

## Overview

The parallel training implementation allows you to:

1. Train multiple difficulty levels simultaneously
2. Improve multiple existing checkpoints in parallel
3. Use all available CPU cores for maximum training speed

The implementation maintains compatibility with existing checkpoints, so you can continue training from your current progress.

## How It Works

The parallel training uses Python's `multiprocessing` module to:
- Create separate processes for each difficulty level being trained
- Each process loads its own model and trains independently
- Results are saved as checkpoints, just like in the original implementation

## Requirements

- Python 3.6+
- PyTorch
- All dependencies from the original implementation

## Usage

### 1. Parallel Training (Progressive)

Train multiple levels in parallel:

```bash
make parallel_train s=1 e=5 r=30 p=4
```

Where:
- `s`: Starting level (scramble moves)
- `e`: Ending level (max level to train)
- `r`: Minimum success rate required
- `p`: Number of processes to use (default: number of CPU cores)

Or use the Python script directly:

```bash
python parallel_cube_rl.py --mode train --level 1 --max_level 5 --min_rate 30 --processes 4
```

### 2. Parallel Improvement

Improve multiple existing checkpoints in parallel:

```bash
make parallel_improve l=1,2,3,4,5 r=90 p=4
```

Where:
- `l`: Comma-separated list of levels to improve
- `r`: Minimum success rate required
- `p`: Number of processes to use

Or use the Python script directly:

```bash
python parallel_cube_rl.py --mode improve --levels 1,2,3,4,5 --min_rate 90 --target_rate 95 --processes 4
```

If no levels are specified, all existing checkpoints will be improved.

### 3. Testing a Model

Test a specific model:

```bash
make parallel_test n=3 t=100
```

Where:
- `n`: Level to test (scramble moves)
- `t`: Number of test cases

## Advanced Configuration

The parallel training supports all the configuration options from the original implementation. Some important ones:

```bash
python parallel_cube_rl.py --mode train \
    --level 1 --max_level 5 \
    --min_episodes 5000 --max_episodes 10000 \
    --target_rate 50 --min_rate 40 \
    --batch_size 64 --use_pregenerated \
    --recent_window 1000 \
    --lr 0.001 --memory_size 100000 \
    --gamma 0.99 --epsilon_start 1.0 \
    --epsilon_min 0.05 --epsilon_decay 0.995 \
    --processes 4
```

## Performance Tips

1. **Process Count**: For best performance, use a number of processes equal to your CPU core count
2. **GPU Usage**: Each process will use the GPU if available, which may cause memory issues with many processes
3. **Memory Usage**: Each process requires separate memory, so monitor system memory usage
4. **Pregenerated Scrambles**: Use `--use_pregenerated` for consistent training across processes

## Examples

### Train levels 1-3 in parallel using 3 processes:

```bash
make parallel_train s=1 e=3 r=50 p=3
```

### Improve levels 2 and 4 with 2 processes:

```bash
make parallel_improve l=2,4 r=90 p=2
```

### Train all levels from 1 to 10 with automatic process allocation:

```bash
make parallel_train s=1 e=10 r=30
``` 
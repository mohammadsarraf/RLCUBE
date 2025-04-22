import os
import sys
import time
import torch
import argparse
import multiprocessing as mp
from functools import partial
import numpy as np

# Import from existing cube_rl module
from cube_rl import (
    train_specific_level, 
    progressive_training,
    test_agent,
    ensure_scrambles_exist,
    find_latest_checkpoint_level
)

def train_level_wrapper(level, max_level, min_episodes, max_episodes, 
                        target_success_rate, min_success_rate, batch_size, 
                        use_pregenerated, recent_window, agent_config, 
                        checkpoint=None, process_id=0):
    """Wrapper function for training a specific level in a separate process"""
    print(f"\n[Process {process_id}] Starting training for level {level}")
    
    # Construct checkpoint path if not provided
    if checkpoint is None and level > 1:
        prev_level = level - 1
        prev_checkpoint = os.path.join("data/modelCheckpoints", f'cube_solver_model_scramble_{prev_level}.pt')
        if os.path.exists(prev_checkpoint):
            checkpoint = prev_checkpoint
            print(f"[Process {process_id}] Using checkpoint from level {prev_level}")
    
    # Train the level
    result_checkpoint = train_specific_level(
        scramble_moves=level,
        min_episodes=min_episodes,
        max_episodes=max_episodes,
        target_success_rate=target_success_rate,
        min_success_rate=min_success_rate,
        batch_size=batch_size,
        prev_checkpoint=checkpoint,
        use_pregenerated=use_pregenerated,
        recent_window=recent_window,
        agent_config=agent_config
    )
    
    print(f"[Process {process_id}] Completed training for level {level}")
    return level, result_checkpoint

def parallel_progressive_training(start_level=None, max_level=5, min_episodes=5000, 
                                max_episodes=10000, target_success_rate=30, 
                                min_success_rate=None, batch_size=64,
                                use_pregenerated=True, custom_checkpoint=None, 
                                recent_window=1000, agent_config=None, 
                                num_processes=None):
    """Train multiple levels in parallel using multiprocessing"""
    
    # Determine number of processes if not specified
    if num_processes is None:
        num_processes = min(mp.cpu_count(), max_level - start_level + 1)
    
    # If no start_level is specified, find the highest checkpoint level
    if start_level is None:
        start_level = find_latest_checkpoint_level() + 1
        if start_level > max_level:
            print(f"Already have checkpoints up to level {start_level-1}, which exceeds max_level={max_level}")
            return
    
    # Make sure we start at least from level 1
    start_level = max(1, start_level)
    
    # Create a list of levels to train
    levels_to_train = list(range(start_level, max_level + 1))
    
    # Adjust number of processes if we have fewer levels
    num_processes = min(num_processes, len(levels_to_train))
    
    print(f"Starting parallel training with {num_processes} processes")
    print(f"Training levels: {levels_to_train}")
    
    # Use a manager to store results that can be shared between processes
    with mp.Manager() as manager:
        # Create a dictionary to store results
        results = manager.dict()
        
        # Create and start processes
        processes = []
        
        for i, level in enumerate(levels_to_train):
            # Determine which checkpoint to use
            checkpoint_to_use = custom_checkpoint
            
            # If this is the first level and no custom checkpoint is provided
            if i == 0 and checkpoint_to_use is None and level > 1:
                prev_level = level - 1
                prev_checkpoint = os.path.join("data/modelCheckpoints", f'cube_solver_model_scramble_{prev_level}.pt')
                if os.path.exists(prev_checkpoint):
                    checkpoint_to_use = prev_checkpoint
            
            # Create process arguments
            process_args = (
                level, max_level, min_episodes, max_episodes,
                target_success_rate, min_success_rate, batch_size,
                use_pregenerated, recent_window, agent_config,
                checkpoint_to_use, i
            )
            
            # Create and start the process
            p = mp.Process(target=train_level_wrapper, args=process_args)
            processes.append(p)
            p.start()
            
            # Slight delay to avoid race conditions in output
            time.sleep(1)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        print("\nAll training processes completed!")

def parallel_improve_levels(levels=None, min_episodes=3000, max_episodes=8000, 
                           target_success_rate=95, min_success_rate=90, batch_size=64,
                           use_pregenerated=True, recent_window=1000, agent_config=None,
                           num_processes=None, plateau_required=5):
    """
    Improve multiple checkpoint levels in parallel
    This function focuses on improving already trained models instead of training from scratch
    """
    # Determine number of processes if not specified
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # If no levels specified, find all available checkpoint levels
    if levels is None:
        levels = []
        for file in os.listdir("data/modelCheckpoints"):
            if file.startswith('cube_solver_model_scramble_') and file.endswith('.pt'):
                try:
                    level = int(file.split('_')[-1].split('.')[0])
                    levels.append(level)
                except ValueError:
                    continue
        
        levels.sort()
    
    if not levels:
        print("No checkpoint levels found to improve.")
        return
    
    # Adjust number of processes if we have fewer levels
    num_processes = min(num_processes, len(levels))
    
    print(f"Starting parallel improvement for levels {levels} with {num_processes} processes")
    print(f"Plateau detection requires {plateau_required} stable measurements")
    
    # Create and start processes
    processes = []
    
    for i, level in enumerate(levels):
        # Get the checkpoint for this level
        checkpoint = os.path.join("data/modelCheckpoints", f'cube_solver_model_scramble_{level}.pt')
        
        if not os.path.exists(checkpoint):
            print(f"Warning: Checkpoint for level {level} not found. Skipping.")
            continue
        
        # Use a more aggressive learning rate for improvement
        improvement_config = agent_config.copy() if agent_config else {}
        improvement_config.update({
            'learning_rate': 0.0005,  # Lower learning rate for fine-tuning
            'epsilon_start': 0.1,     # Lower initial exploration
            'epsilon_min': 0.01,      # Lower minimum exploration
            'epsilon_decay': 0.998,   # Slower decay
            'plateau_required': plateau_required  # Add the plateau parameter
        })
        
        # Create process arguments
        process_args = (
            level, level, min_episodes, max_episodes,
            target_success_rate, min_success_rate, batch_size,
            use_pregenerated, recent_window, improvement_config,
            checkpoint, i
        )
        
        # Create and start the process
        p = mp.Process(target=train_level_wrapper, args=process_args)
        processes.append(p)
        p.start()
        
        # Slight delay to avoid race conditions in output
        time.sleep(1)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("\nAll improvement processes completed!")

def main():
    """Main function for parallel training"""
    parser = argparse.ArgumentParser(description='Train a Rubik\'s Cube solving agent in parallel')
    parser.add_argument('--mode', type=str, choices=['train', 'improve', 'test'], default='train',
                        help='Operation mode: train (progressive training), improve (existing checkpoints), or test')
    parser.add_argument('--level', type=int, default=None, 
                        help='Starting difficulty level (scramble moves). If not provided, will start from next level after highest checkpoint')
    parser.add_argument('--max_level', type=int, default=5, 
                        help='Maximum difficulty level to train up to')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated list of levels to improve in improve mode (e.g., "1,2,3")')
    parser.add_argument('--min_episodes', type=int, default=5000, 
                        help='Minimum episodes per difficulty level')
    parser.add_argument('--max_episodes', type=int, default=10000, 
                        help='Maximum episodes per difficulty level')
    parser.add_argument('--target_rate', type=int, default=50, 
                        help='Target success rate to achieve before moving to next level')
    parser.add_argument('--min_rate', type=int, default=None, 
                        help='Minimum success rate to achieve regardless of episode count')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training')
    parser.add_argument('--use_pregenerated', action='store_true',
                        help='Use pregenerated scrambles from nMovescramble.txt instead of random scrambles')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a specific model checkpoint to use for training')
    parser.add_argument('--recent_window', type=int, default=1000,
                        help='Number of recent episodes to consider for calculating success rate')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of parallel processes to use (default: number of CPU cores)')
    parser.add_argument('--test_level', type=int, default=None,
                        help='Level to test in test mode')
    parser.add_argument('--num_tests', type=int, default=100,
                        help='Number of test cases in test mode')
    
    # DQN Agent configuration arguments
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the DQN agent')
    parser.add_argument('--memory_size', type=int, default=100000,
                        help='Memory size for experience replay')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for future rewards')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='Starting exploration rate')
    parser.add_argument('--epsilon_min', type=float, default=0.05,
                        help='Minimum exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                        help='Decay rate for exploration')
    parser.add_argument('--target_update', type=int, default=1000,
                        help='How often to update target network (steps)')
    
    # Plateau detection parameter
    parser.add_argument('--plat', type=int, default=5,
                        help='Number of stable measurements required to detect a plateau')
    
    # Advanced DQN features
    parser.add_argument('--no_double_dqn', action='store_false', dest='double_dqn',
                        help='Disable Double DQN (use regular DQN)')
    parser.add_argument('--no_dueling_dqn', action='store_false', dest='dueling_dqn',
                        help='Disable Dueling DQN architecture')
    parser.add_argument('--no_prioritized', action='store_false', dest='prioritized_replay',
                        help='Disable prioritized experience replay')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Priority exponent (alpha) for prioritized replay')
    parser.add_argument('--beta', type=float, default=0.4,
                        help='Initial importance sampling weight (beta) for prioritized replay')
    
    # Set defaults for advanced features
    parser.set_defaults(double_dqn=True, dueling_dqn=True, prioritized_replay=True)
    
    args = parser.parse_args()
    
    # Build agent configuration
    agent_config = {
        'learning_rate': args.lr,
        'memory_size': args.memory_size,
        'gamma': args.gamma,
        'epsilon_start': args.epsilon_start,
        'epsilon_min': args.epsilon_min,
        'epsilon_decay': args.epsilon_decay,
        'target_update_freq': args.target_update,
        'double_dqn': args.double_dqn,
        'dueling_dqn': args.dueling_dqn,
        'prioritized_replay': args.prioritized_replay,
        'alpha': args.alpha,
        'beta': args.beta
    }
    
    # Execute based on mode
    if args.mode == 'train':
        parallel_progressive_training(
            start_level=args.level,
            max_level=args.max_level,
            min_episodes=args.min_episodes,
            max_episodes=args.max_episodes,
            target_success_rate=args.target_rate,
            min_success_rate=args.min_rate,
            batch_size=args.batch_size,
            use_pregenerated=args.use_pregenerated,
            custom_checkpoint=args.model,
            recent_window=args.recent_window,
            agent_config=agent_config,
            num_processes=args.processes
        )
    elif args.mode == 'improve':
        levels = [int(level) for level in args.levels.split(',')] if args.levels else None
        
        parallel_improve_levels(
            levels=levels,
            min_episodes=args.min_episodes,
            max_episodes=args.max_episodes,
            target_success_rate=args.target_rate,
            min_success_rate=args.min_rate,
            batch_size=args.batch_size,
            use_pregenerated=args.use_pregenerated,
            recent_window=args.recent_window,
            agent_config=agent_config,
            num_processes=args.processes,
            plateau_required=args.plat
        )
    elif args.mode == 'test':
        if args.test_level is None:
            print("Error: --test_level must be specified with test mode")
            sys.exit(1)
        if args.model is None:
            checkpoint = os.path.join("data/modelCheckpoints", f'cube_solver_model_scramble_{args.test_level}.pt')
            if not os.path.exists(checkpoint):
                print(f"Error: No checkpoint found for level {args.test_level}. Please specify --model.")
                sys.exit(1)
        else:
            checkpoint = args.model
            
        test_agent(
            num_tests=args.num_tests, 
            scramble_moves=args.test_level, 
            checkpoint_path=checkpoint, 
            use_pregenerated=args.use_pregenerated
        )

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for compatibility with CUDA
    # This is important for PyTorch with CUDA on Windows
    mp.set_start_method('spawn', force=True)
    main() 
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import json
import subprocess
import time
import rl_agent

def load_scrambles(cube, num_moves):
    """Load pregenerated scrambles from file"""
    cube.pregenerated_scrambles = []
    os.makedirs("data/scrambles", exist_ok=True)  # Create scrambles directory if it doesn't exist
    filepath = os.path.join("data/scrambles", f"{num_moves}movescramble.txt")

    # Check if scramble file exists, generate if it doesn't
    if not os.path.exists(filepath) and cube.use_pregenerated:
        print(f"Scramble file {filepath} not found. Generating now...")
        try:
            subprocess.run(["python", "src/scramble_generator.py", "--level", str(num_moves)], check=True)
        except Exception as e:
            print(f"Error generating scrambles: {e}")
            cube.use_pregenerated = False
            return

    # Load scrambles from file  
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            cube.pregenerated_scrambles = [json.loads(line.strip()) for line in f if line.strip()]
        if not cube.pregenerated_scrambles:
            print("No valid scrambles found, switching to random scrambles")
            cube.use_pregenerated = False
    else:
        print(f"Warning: Scramble file {filepath} not found. Will use random scrambles instead.")
        cube.use_pregenerated = False

    print(f"Loaded {len(cube.pregenerated_scrambles)} pregenerated scrambles with {num_moves} solution length")

def check_and_regenerate_scrambles(cube):
    """Check if we're running out of scrambles and regenerate if needed"""
    if cube.use_pregenerated and cube.pregenerated_scrambles:
        cube.scramble_usage_count += 1
        
        # Regenerate when we've used all scrambles
        if cube.scramble_usage_count >= len(cube.pregenerated_scrambles):
            print(f"\nUsed all {len(cube.pregenerated_scrambles)} scrambles. Generating a new set...")
            try:
                subprocess.run(["python", "src/scramble_generator.py", "--level", str(cube.scramble_moves)], check=True)
                cube.scramble_usage_count = 0  # Reset counter
                load_scrambles(cube, cube.scramble_moves)  # Reload scrambles
                print(f"Successfully regenerated scrambles for level {cube.scramble_moves}")
            except Exception as e:
                print(f"Error regenerating scrambles: {e}")

def find_latest_checkpoint_level():
    """Find the highest difficulty level with an existing checkpoint"""
    max_level = 0
    for file in os.listdir("data/modelCheckpoints"):
        if file.startswith('cube_solver_model_scramble_') and file.endswith('.pt'):
            try:
                level = int(file.split('_')[-1].split('.')[0])  # Extract scramble level from filename
                max_level = max(max_level, level)
            except ValueError:
                continue
    return max_level

def ensure_scrambles_exist(scramble_moves, use_pregenerated=False):
    """
    Make sure scramble files exist for the given difficulty level.
    Returns True if scrambles exist or were successfully generated, False otherwise.
    """
    if not use_pregenerated:
        return True  # No need to check if not using pregenerated scrambles
        
    filepath = os.path.join("data/scrambles", f"{scramble_moves}movescramble.txt")
    if os.path.exists(filepath) and open(filepath).readline().strip():  # Check if file has content
        return True

    # File doesn't exist or is empty, generate scrambles
    print(f"Generating scrambles for difficulty level {scramble_moves}...")
    try:
        subprocess.run(["python", "src/scramble_generator.py", "--level", str(scramble_moves)], check=True)
        return os.path.exists(filepath)  # Return True if scramble file is generated
    except Exception as e:
        print(f"Error generating scrambles for level {scramble_moves}: {e}")
        return False

def log_training_results(scramble_moves, episode, solved_episodes, total_episodes, success_rate, recent_success_rate, time_taken, agent_config=None, level_stats=None, recent_scrambles=None, avg_reward=None, avg_loss=None, epsilon=None, memory_size=None, best_success_rate=None, episodes_since_improvement=None, stopping_reason=None):
    """Log training results to a file in the output directory"""
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Create timestamp for the log file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = f"output/training_log_{scramble_moves}moves_{timestamp}.txt"
    
    # Format the log content
    log_content = f"""
=== Training Results ===
Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}
Difficulty Level: {scramble_moves} moves
Episodes: {episode}
Solved Episodes: {solved_episodes}/{total_episodes}
Success Rate: {success_rate:.2f}%
Recent Success Rate: {recent_success_rate:.2f}%
Training Time: {time_taken:.2f} seconds
"""
    
    # Add stopping reason if available
    if stopping_reason:
        log_content += f"Stopping Reason: {stopping_reason}\n"
    
    # Add performance metrics if available
    if avg_reward is not None:
        log_content += f"Average Reward: {avg_reward:.2f}\n"
    if avg_loss is not None:
        log_content += f"Average Loss: {avg_loss:.4f}\n"
    if epsilon is not None:
        log_content += f"Exploration Rate (Epsilon): {epsilon:.4f}\n"
    if memory_size is not None:
        log_content += f"Memory Size: {memory_size}\n"
    if best_success_rate is not None:
        log_content += f"Best Success Rate: {best_success_rate:.2f}%\n"
    if episodes_since_improvement is not None:
        log_content += f"Episodes Since Last Improvement: {episodes_since_improvement}\n"
    
    # Add level statistics if available
    if level_stats:
        log_content += "\nLevel Statistics:\n"
        for stat in level_stats:
            log_content += f"{stat}\n"
    
    # Add recent scrambles if available
    if recent_scrambles:
        log_content += f"\nLast {len(recent_scrambles)} Scrambles:\n"
        for i, (scr, out, mvs, rew) in enumerate(recent_scrambles):
            log_content += f"- {scr} ({out} in {mvs} moves, reward: {rew:.1f})\n"
    
    # Add agent configuration if provided
    if agent_config:
        log_content += "\nAgent Configuration:\n"
        for key, value in agent_config.items():
            log_content += f"  - {key}: {value}\n"
    
    # Write to log file
    with open(log_file, "w") as f:
        f.write(log_content)
    
    print(f"\nTraining results logged to: {log_file}")
    return log_file

def train_specific_level(scramble_moves, min_episodes=5000, max_episodes=10000, 
                         target_success_rate=30, min_success_rate=None, batch_size=64, prev_checkpoint=None, 
                         use_pregenerated=False, recent_window=1000, agent_config=None):
    """
    Train on a specific difficulty level
    
    Args:
        scramble_moves: Number of scramble moves for this difficulty level
        min_episodes: Minimum number of episodes to train for
        max_episodes: Maximum number of episodes to train for
        target_success_rate: Target success rate to achieve before stopping training
        min_success_rate: Minimum success rate to achieve regardless of episode count
        batch_size: Batch size for training
        prev_checkpoint: Path to previous checkpoint to load
        use_pregenerated: Whether to use pregenerated scrambles
        recent_window: Number of recent episodes to consider for success rate calculation
        agent_config: Dictionary of DQNAgent configuration parameters
    """
    print(f"\n=== Starting training with {scramble_moves} scramble moves ===")
    
    # If min_success_rate is not specified, use target_success_rate
    if min_success_rate is None:
        min_success_rate = target_success_rate
    
    # For very high target rates (>90%), disable early stopping to ensure we reach the target
    disable_early_stopping = target_success_rate >= 90

    # If target rate is extremely high (99%+), use a small value for better precision
    if target_success_rate >= 99:
        min_success_rate = target_success_rate - 0.5  # Within 0.5% is acceptable
        print(f"High target success rate detected ({target_success_rate}%). Training will continue until reaching at least {min_success_rate}%")
        if disable_early_stopping:
            print("Early stopping disabled for high target success rate.")
    
    # Ensure scrambles exist if using pregenerated scrambles
    if use_pregenerated:
        if not ensure_scrambles_exist(scramble_moves, use_pregenerated):
            print(f"Warning: Could not generate scrambles for level {scramble_moves}. Disabling pregenerated scrambles.")
            use_pregenerated = False
    
    # Create the primary environment with current difficulty
    main_env = rl_agent.CubeEnvironment(scramble_moves=scramble_moves, use_pregenerated=use_pregenerated)
    
    # Create a set of environments with all difficulties up to current
    all_envs = {}
    for i in range(1, scramble_moves + 1):
        # Ensure scrambles exist for each difficulty level we'll use
        if use_pregenerated and not ensure_scrambles_exist(i, use_pregenerated):
            print(f"Warning: Could not generate scrambles for level {i}. Using random scrambles for this level.")
            all_envs[i] = rl_agent.CubeEnvironment(scramble_moves=i, use_pregenerated=False)
        else:
            all_envs[i] = rl_agent.CubeEnvironment(scramble_moves=i, use_pregenerated=use_pregenerated)
    
    state_size = 6 * 9 * 6  # 6 faces, 9 stickers per face, 6 possible colors
    action_size = len(rl_agent.MOVES)
    
    # Set default agent configuration if not provided
    default_config = {
        'learning_rate': 0.001,
        'memory_size': 100000,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.995,
        'target_update_freq': 1000,
        'double_dqn': True,
        'dueling_dqn': True,
        'prioritized_replay': True,
        'alpha': 0.6,
        'beta': 0.4,
        'beta_increment': 0.001,
        'plateau_required': 5  # Default value for plateau detection
    }
    
    # Override defaults with provided configuration
    if agent_config:
        default_config.update(agent_config)
    
    # Extract plateau detection parameter
    plateau_required = default_config.get('plateau_required', 5)
    print(f"Plateau detection will require {plateau_required} stable measurements")
    
    # Initialize agent with configuration
    agent = rl_agent.DQNAgent(
        state_size=state_size, 
        action_size=action_size,
        learning_rate=default_config['learning_rate'],
        memory_size=default_config['memory_size'],
        gamma=default_config['gamma'],
        epsilon_start=default_config['epsilon_start'],
        epsilon_min=default_config['epsilon_min'],
        epsilon_decay=default_config['epsilon_decay'],
        target_update_freq=default_config['target_update_freq'],
        double_dqn=default_config['double_dqn'],
        dueling_dqn=default_config['dueling_dqn'],
        prioritized_replay=default_config['prioritized_replay'],
        alpha=default_config['alpha'],
        beta=default_config['beta'],
        beta_increment=default_config['beta_increment']
    )
    
    # Print agent configuration
    print(f"Agent Configuration:")
    for key, value in default_config.items():
        print(f"  - {key}: {value}")
    
    # Load previous checkpoint if available
    if prev_checkpoint:
        try:
            agent.policy_net.load_state_dict(torch.load(prev_checkpoint))
            # Also load to target network
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"Loaded checkpoint from {prev_checkpoint}")
            
            # Use a lower epsilon when using a checkpoint to better utilize learned knowledge
            # For the same level, use very little exploration
            if os.path.basename(prev_checkpoint) == f'cube_solver_model_scramble_{scramble_moves}.pt':
                agent.epsilon = max(0.05, agent.epsilon_min)  # Minimal exploration when continuing same level
                print(f"Continuing training on same level {scramble_moves} with minimal exploration (epsilon: {agent.epsilon:.4f})")
            else:
                # For a new difficulty level, use moderate exploration 
                agent.epsilon = 0.2  # 20% exploration when moving to a new level
                print(f"Training level {scramble_moves} from previous level checkpoint with moderate exploration (epsilon: {agent.epsilon:.4f})")
        except Exception as e:
            print(f"Failed to load checkpoint from {prev_checkpoint}, starting fresh: {e}")
    
    solved_episodes = 0
    solved_by_difficulty = {i: 0 for i in range(1, scramble_moves + 1)}
    episodes_by_difficulty = {i: 0 for i in range(1, scramble_moves + 1)}
    start_time = time.time()
    current_success_rate = 0
    recent_success_rate = 0
    episode = 0
    
    # Keep track of recent scrambles and outcomes
    recent_scrambles = []  # Will store tuples of (scramble, outcome, moves)
    
    # Keep track of recent episodes for success rate calculation
    recent_episodes = deque(maxlen=recent_window)
    recent_solved = deque(maxlen=recent_window)
    
    # Track training metrics
    training_losses = []
    avg_rewards = deque(maxlen=100)
    
    # Define the final checkpoint name for this difficulty level
    final_checkpoint = os.path.join("data/modelCheckpoints", f'cube_solver_model_scramble_{scramble_moves}.pt')
    best_checkpoint = os.path.join("data/modelCheckpoints", f'cube_solver_model_scramble_{scramble_moves}_best.pt')
    
    # Variables for tracking best model
    best_success_rate = 0
    episodes_since_improvement = 0
    patience = 500  # Number of episodes to wait for improvement before early stopping
    
    # Smart plateau detection parameters
    success_rate_history = []  # Track success rates over time
    plateau_window = 10  # Check last 10 chunks of 500 episodes each
    plateau_chunk_size = 500  # Each chunk represents 500 episodes
    plateau_threshold = 0.3  # Required improvement percentage points between chunks
    stable_rate_count = 0  # Count of stable measurements
    stable_rate_required = plateau_required  # Use the configurable parameter
    
    # Training loop - continue until we reach target success rate or max episodes
    # Also ensure we continue training if we haven't reached min_success_rate
    while (episode < max_episodes or (recent_success_rate < min_success_rate)):
        episode += 1
        
        # Early stopping if no improvement for a long time and not disabled
        if not disable_early_stopping and episodes_since_improvement > patience and episode > min_episodes:
            print(f"No improvement for {patience} episodes. Early stopping at episode {episode}.")
            print(f"Best success rate achieved: {best_success_rate:.2f}% (target was {target_success_rate}%)")
            break
            
        # Distribution of difficulties:
        # 70% current difficulty, 30% distributed among easier difficulties
        if scramble_moves == 1 or random.random() < 0.7:
            # Use current difficulty most of the time
            selected_diff = scramble_moves
        else:
            # Select from easier difficulties with preference toward harder ones
            weights = [i/sum(range(1, scramble_moves)) for i in range(1, scramble_moves)]
            selected_diff = random.choices(range(1, scramble_moves), weights=weights)[0]
        
        # Use the environment with selected difficulty
        env_to_use = all_envs[selected_diff]
        episodes_by_difficulty[selected_diff] += 1
        
        state = env_to_use.reset()
        done = False
        moves_taken = 0
        episode_reward = 0
        
        # Get the scramble that was applied
        scramble = " ".join(env_to_use.scramble_sequence) if env_to_use.scramble_sequence else "Random scramble"
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env_to_use.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            moves_taken += 1
            episode_reward += reward
            
            if done:
                solved = env_to_use.cube.is_solved()
                outcome = "Solved" if solved else "Failed"
                # Add to recent scrambles
                recent_scrambles.append((scramble, outcome, moves_taken, episode_reward))
                # Keep only the 5 most recent scrambles
                if len(recent_scrambles) > 5:
                    recent_scrambles.pop(0)
                
                if solved:
                    solved_by_difficulty[selected_diff] += 1
                    if selected_diff == scramble_moves:  # Count solutions for current difficulty
                        solved_episodes += 1
                        
                # Track recent episodes (only for the current difficulty level)
                if selected_diff == scramble_moves:
                    recent_episodes.append(1)
                    recent_solved.append(1 if solved else 0)
                    avg_rewards.append(episode_reward)
                break
        
        # Train the agent with experiences from memory
        loss = agent.replay(batch_size)
        training_losses.append(loss)
        
        # Print progress and check success rate every 100 episodes
        if episode % 100 == 0 or episode == 1:
            # Calculate success rate for current difficulty
            main_episodes = episodes_by_difficulty[scramble_moves]
            main_solved = solved_by_difficulty[scramble_moves]
            
            if main_episodes > 0:
                current_success_rate = (main_solved / main_episodes) * 100
            
            # Calculate recent success rate
            if len(recent_episodes) > 0:
                recent_success_rate = (sum(recent_solved) / len(recent_episodes)) * 100
            else:
                recent_success_rate = 0
            
            # Calculate average reward
            avg_reward = sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0
            
            all_solved = sum(solved_by_difficulty.values())
            all_episodes = sum(episodes_by_difficulty.values())
            overall_rate = (all_solved / all_episodes) * 100
            
            # Update best model if improved
            if recent_success_rate > best_success_rate and len(recent_episodes) >= 200:
                best_success_rate = recent_success_rate
                torch.save(agent.policy_net.state_dict(), best_checkpoint)
                print(f"\nNew best model saved with success rate: {best_success_rate:.2f}%")
                episodes_since_improvement = 0
            else:
                episodes_since_improvement += 100
            
            # Store success rate history for plateau detection (every 500 episodes)
            if episode % 500 == 0 and len(recent_episodes) >= 300:
                success_rate_history.append(recent_success_rate)
                
                # Only start checking for plateau after we have enough data points
                if len(success_rate_history) >= plateau_window:
                    # Keep only the most recent measurements
                    if len(success_rate_history) > plateau_window:
                        success_rate_history = success_rate_history[-plateau_window:]
                    
                    # Check if success rate has plateaued
                    improving = False
                    
                    # Calculate average improvement over the last several chunks
                    improvements = [success_rate_history[i] - success_rate_history[i-1] 
                                   for i in range(1, len(success_rate_history))]
                    avg_improvement = sum(improvements) / len(improvements)
                    
                    # Check the trend in the last few measurements
                    recent_improvements = improvements[-3:] if len(improvements) >= 3 else improvements
                    recent_avg_improvement = sum(recent_improvements) / len(recent_improvements)
                    
                    # Detect if we're improving at a meaningful rate
                    if recent_avg_improvement >= plateau_threshold or avg_improvement >= plateau_threshold * 0.7:
                        improving = True
                        stable_rate_count = 0  # Reset stability counter if still improving
                    
                    # If we've reached a good success rate and improvement has plateaued
                    if not improving and recent_success_rate >= 50:  # Only consider plateau if success rate is decent
                        stable_rate_count += 1
                        
                        # Log plateau detection progress
                        remaining = stable_rate_required - stable_rate_count
                        if remaining > 0:
                            print(f"\nPlateau detection: Success rate has stabilized around {recent_success_rate:.2f}%")
                            print(f"Recent improvement rate: {recent_avg_improvement:.2f}% per {plateau_chunk_size} episodes")
                            print(f"Will confirm plateau in {remaining} more measurement{'s' if remaining > 1 else ''}")
                        
                        # If we've confirmed a plateau and no significant improvement for a while
                        if stable_rate_count >= stable_rate_required:
                            # If we're already at a good success rate, consider this the maximum achievable
                            if recent_success_rate >= 75:  # 75% is generally a good success rate
                                print(f"\n=== PLATEAU DETECTED ===")
                                print(f"Success rate has stabilized at {recent_success_rate:.2f}% after {episode} episodes")
                                print(f"This appears to be the maximum achievable rate for level {scramble_moves}")
                                print(f"Target rate of {target_success_rate}% may not be achievable for this difficulty")
                                print(f"Stopping training and saving best model with success rate: {best_success_rate:.2f}%")
                                
                                # Update the log file to indicate why we stopped
                                plateau_reason = f"Training stopped due to plateau detection. Maximum achievable rate appears to be ~{recent_success_rate:.2f}%"
                                
                                break
                    else:
                        stable_rate_count = 0  # Reset if we're still improving or success rate is too low
            
            # Clear screen for cleaner display
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Display training progress
            print(f"=== Training Progress === - Level: {scramble_moves}")
            print(f"  - Episode: {episode}/{max_episodes if episode <= max_episodes else 'unlimited until '+str(min_success_rate)+'% success'}")
            print(f"  - Min Episodes: {min_episodes} - Max Episodes: {max_episodes}")
            print(f"  - Target Success Rate: {target_success_rate}% - Min Success Rate: {min_success_rate}%")
            print(f"  - Batch Size: {batch_size} - Using Pregenerated Scrambles: {use_pregenerated}")
            print(f"  - Previous Checkpoint: {prev_checkpoint if prev_checkpoint else 'None'}")
            print("=" * 50)
            
            # Performance metrics
            print(f"Performance Summary:")
            print(f"✓ Overall Success Rate: {current_success_rate:.2f}%")
            print(f"✓ Recent Success Rate (Last {len(recent_episodes)} Episodes): {recent_success_rate:.2f}%")
            print(f"✓ Best Success Rate: {best_success_rate:.2f}%") 
            print(f"✓ Average Reward: {avg_reward:.2f}")
            print(f"✓ Average Loss: {sum(training_losses[-100:]) / len(training_losses[-100:]) if training_losses else 0:.4f}")
            print(f"✓ Exploration Rate (Epsilon): {agent.epsilon:.4f}")
            print(f"✓ Memory Size: {agent.get_memory_length()}")
            print(f"✓ Episodes Since Last Improvement: {episodes_since_improvement}")
            print()
            
            # Level statistics
            print(f"Level Statistics:")
            diff_stats = [f"D{d}: {solved_by_difficulty[d]}/{episodes_by_difficulty[d]} "
                         f"({(solved_by_difficulty[d]/episodes_by_difficulty[d])*100:.1f}%)" 
                         for d in sorted(solved_by_difficulty.keys()) 
                         if episodes_by_difficulty[d] > 0]
            for stat in diff_stats:
                print(stat)
            print()
            
            # Recent scrambles
            print(f"Last {len(recent_scrambles)} Scrambles:")
            for i, (scr, out, mvs, rew) in enumerate(recent_scrambles):
                print(f"- {scr} ({out} in {mvs} moves, reward: {rew:.1f})")
                
            # Save intermediate checkpoint, replacing the previous one
            if episode % 1000 == 0:
                torch.save(agent.policy_net.state_dict(), final_checkpoint)
                print(f"\nSaved checkpoint: {final_checkpoint}")
        
        # Check if we've met the success criteria and minimum episodes
        # Use recent success rate instead of overall success rate
        # And we must have enough episodes to have a meaningful measurement
        if episode >= min_episodes and len(recent_episodes) >= min(recent_window, 500):
            if recent_success_rate >= target_success_rate:
                print(f"\nReached target success rate of {target_success_rate}% (actual {recent_success_rate:.2f}%) after {episode} episodes!")
                break
            elif recent_success_rate >= min_success_rate and best_success_rate >= target_success_rate:
                # If we've reached the minimum required rate and previously hit the target rate
                print(f"\nReached minimum success rate of {min_success_rate}% with best at {best_success_rate:.2f}% (target: {target_success_rate}%) after {episode} episodes!")
                break
    
    # Save final model for this scramble difficulty
    torch.save(agent.policy_net.state_dict(), final_checkpoint)
    total_time = time.time() - start_time
    
    # Determine stopping reason for logging
    if 'plateau_reason' in locals():
        stopping_reason = plateau_reason
    elif recent_success_rate >= target_success_rate:
        stopping_reason = f"Reached target success rate of {target_success_rate}%"
    elif episodes_since_improvement > patience and not disable_early_stopping:
        stopping_reason = f"Early stopping due to no improvement for {patience} episodes"
    elif episode >= max_episodes:
        stopping_reason = f"Reached maximum number of episodes: {max_episodes}"
    else:
        stopping_reason = "Training completed normally"
    
    print(f"\nTraining completed. Scramble moves: {scramble_moves}, "
          f"Episodes: {episode}, Solved: {solved_episodes}/{episodes_by_difficulty[scramble_moves]}, "
          f"Success Rate: {current_success_rate:.2f}%, Recent Success Rate: {recent_success_rate:.2f}%, "
          f"Time: {total_time:.2f} seconds.")
    print(f"Stopping reason: {stopping_reason}")
    
    # Calculate level statistics for logging
    level_stats = [f"D{d}: {solved_by_difficulty[d]}/{episodes_by_difficulty[d]} "
                   f"({(solved_by_difficulty[d]/episodes_by_difficulty[d])*100:.1f}%)" 
                   for d in sorted(solved_by_difficulty.keys()) 
                   if episodes_by_difficulty[d] > 0]
    
    # Calculate average loss for logging
    avg_loss = sum(training_losses[-100:]) / len(training_losses[-100:]) if training_losses else 0
    
    # Calculate average reward for logging
    avg_reward = sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0
    
    # Log training results to file
    log_file = log_training_results(
        scramble_moves=scramble_moves,
        episode=episode,
        solved_episodes=solved_episodes,
        total_episodes=episodes_by_difficulty[scramble_moves],
        success_rate=current_success_rate,
        recent_success_rate=recent_success_rate,
        time_taken=total_time,
        agent_config=default_config,
        level_stats=level_stats,
        recent_scrambles=recent_scrambles,
        avg_reward=avg_reward,
        avg_loss=avg_loss,
        epsilon=agent.epsilon,
        memory_size=agent.get_memory_length(),
        best_success_rate=best_success_rate,
        episodes_since_improvement=episodes_since_improvement,
        stopping_reason=stopping_reason
    )
    
    # Use the best model if we have one
    if os.path.exists(best_checkpoint) and best_success_rate > current_success_rate:
        print(f"Loading best model with success rate {best_success_rate:.2f}% for testing")
        agent.policy_net.load_state_dict(torch.load(best_checkpoint))
        # Copy best model to final checkpoint
        torch.save(agent.policy_net.state_dict(), final_checkpoint)
    
    # Test on current difficulty
    print(f"\n=== Testing agent with {scramble_moves} scramble moves ===")
    test_agent(num_tests=50, scramble_moves=scramble_moves, checkpoint_path=final_checkpoint, use_pregenerated=use_pregenerated)
    
    return final_checkpoint

def progressive_training(start_level=None, max_scramble=20, min_episodes=5000, 
                         max_episodes=10000, target_success_rate=30, min_success_rate=None, batch_size=64,
                         use_pregenerated=True, custom_checkpoint=None, recent_window=1000, agent_config=None):
    """Train progressively with increasing scramble difficulty
    
    Args:
        start_level: Starting difficulty level (scramble moves). If None, continue from highest checkpoint.
        max_scramble: Maximum difficulty level to train up to
        min_episodes: Minimum episodes per difficulty level
        max_episodes: Maximum episodes per difficulty level
        target_success_rate: Target success rate to achieve before moving to next level
        min_success_rate: Minimum success rate to achieve regardless of episode count
        batch_size: Batch size for training
        use_pregenerated: Whether to use pregenerated scrambles
        custom_checkpoint: Path to a specific checkpoint to use instead of searching for latest
        recent_window: Number of recent episodes to consider for success rate calculation
        agent_config: Dictionary of DQNAgent configuration parameters
    """
    
    # If no start_level is specified, try to find the highest level with a checkpoint
    if start_level is None:
        start_level = find_latest_checkpoint_level() + 1
        if start_level > max_scramble:
            print(f"Already have checkpoints up to level {start_level-1}, which exceeds max_scramble={max_scramble}")
            return
        
    # Make sure we start at least from level 1
    start_level = max(1, start_level)
    print(f"Starting progressive training from level {start_level} up to {max_scramble}")
    
    checkpoint = custom_checkpoint if custom_checkpoint else None
    
    # If no custom checkpoint was provided and we're starting beyond level 1, 
    # load the checkpoint from the previous level
    if checkpoint is None and start_level > 1:
        prev_level = start_level - 1
        prev_checkpoint = os.path.join("data/modelCheckpoints", f'cube_solver_model_scramble_{prev_level}.pt')
        if os.path.exists(prev_checkpoint):
            checkpoint = prev_checkpoint
            print(f"Will use checkpoint from level {prev_level}: {prev_checkpoint}")
        else:
            print(f"Warning: Starting at level {start_level} but no checkpoint found for level {prev_level}")
    elif checkpoint:
        print(f"Using custom checkpoint: {checkpoint}")
    
    # Print agent configuration if provided
    if agent_config:
        print("Using custom agent configuration:")
        for key, value in agent_config.items():
            print(f"  - {key}: {value}")
            
    # Train progressively from start_level to max_scramble
    for scramble_moves in range(start_level, max_scramble + 1):
        # For each difficulty level, customize agent config if needed
        current_agent_config = agent_config.copy() if agent_config else {}
        
        # You can customize config per level if needed, for example:
        # if scramble_moves > 5:
        #    current_agent_config['epsilon_decay'] = 0.997  # Slower decay for harder levels
        
        checkpoint = train_specific_level(
            scramble_moves=scramble_moves,
            min_episodes=min_episodes,
            max_episodes=max_episodes,
            target_success_rate=target_success_rate,
            min_success_rate=min_success_rate,
            batch_size=batch_size,
            prev_checkpoint=checkpoint,
            use_pregenerated=use_pregenerated,
            recent_window=recent_window,
            agent_config=current_agent_config
        )

def test_agent(num_tests=100, scramble_moves=1, checkpoint_path=None, use_pregenerated=True):
    """Test a trained agent on cube solving"""
    # Ensure scrambles exist if using pregenerated scrambles
    if use_pregenerated:
        if not ensure_scrambles_exist(scramble_moves, use_pregenerated):
            print(f"Warning: Could not generate scrambles for testing level {scramble_moves}. Using random scrambles.")
            use_pregenerated = False
    
    env = rl_agent.CubeEnvironment(scramble_moves=scramble_moves, use_pregenerated=use_pregenerated)
    state_size = 6 * 9 * 6
    action_size = len(rl_agent.MOVES)
    
    # Initialize agent with default configuration for testing
    # For testing we don't need some features like prioritized replay
    agent = rl_agent.DQNAgent(
        state_size=state_size, 
        action_size=action_size,
        epsilon_start=0.0,  # No exploration during testing
        epsilon_min=0.0,
        prioritized_replay=False  # Not needed for testing
    )
    
    # Load checkpoint if provided
    if checkpoint_path:
        try:
            agent.policy_net.load_state_dict(torch.load(checkpoint_path))
            # Also load to target network (though we won't use it for testing)
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"Testing with checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            return
    
    # Set epsilon to minimum for best performance (minimal exploration)
    agent.epsilon = 0.0  # No exploration for testing
    
    solved_count = 0
    total_moves = 0
    
    # Track recent test results
    recent_tests = []
    
    # Track all solution moves for statistics
    all_solution_moves = []
    
    for test in range(num_tests):
        state = env.reset()
        done = False
        moves = 0
        
        # Get the scramble that was applied
        scramble = " ".join(env.scramble_sequence) if env.scramble_sequence else "Random scramble"
        
        while not done and moves < env.max_steps:
            action = agent.act(state)
            state, _, done = env.step(action)
            moves += 1
            
            if done and env.cube.is_solved():
                solved_count += 1
                total_moves += moves
                all_solution_moves.append(moves)
                recent_tests.append((test+1, scramble, "Solved", moves))
                break
        
        if not done or not env.cube.is_solved():
            recent_tests.append((test+1, scramble, "Failed", moves))
        
        # Keep only the most recent 10 tests
        if len(recent_tests) > 10:
            recent_tests.pop(0)
        
        # Update display every few tests
        if (test + 1) % 5 == 0 or test == 0 or test == num_tests - 1:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Display test progress
            print(f"=== Testing Progress === ")
            print(f"Testing Level: {scramble_moves}")
            print(f"Using Checkpoint: {checkpoint_path}")
            print(f"Test: {test+1}/{num_tests}")
            print()
            
            # Current statistics
            current_success_rate = (solved_count / (test+1)) * 100
            current_avg_moves = total_moves / solved_count if solved_count > 0 else 0
            
            print(f"Current Statistics:")
            print(f"✓ Success Rate: {current_success_rate:.2f}%")
            print(f"✓ Average Moves: {current_avg_moves:.2f}")
            print(f"✓ Solved: {solved_count}/{test+1}")
            
            # Calculate median and percentiles if we have solved cases
            if all_solution_moves:
                median_moves = sorted(all_solution_moves)[len(all_solution_moves)//2]
                p25_moves = sorted(all_solution_moves)[int(len(all_solution_moves)*0.25)]
                p75_moves = sorted(all_solution_moves)[int(len(all_solution_moves)*0.75)]
                min_moves = min(all_solution_moves)
                max_moves = max(all_solution_moves)
                print(f"✓ Move Statistics: min={min_moves}, 25%={p25_moves}, median={median_moves}, 75%={p75_moves}, max={max_moves}")
            
            print()
            
            # Recent test results
            print(f"Recent Test Results:")
            for test_num, scr, result, mv in recent_tests:
                print(f"- Test {test_num}: {scr} ({result} in {mv} moves)")
    
    # Final statistics
    success_rate = (solved_count / num_tests) * 100
    avg_moves = total_moves / solved_count if solved_count > 0 else 0
    
    print(f"\nFinal Test Results:")
    print(f"Scramble Moves: {scramble_moves}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Moves for Successful Solves: {avg_moves:.2f}")
    
    # More detailed move statistics
    if all_solution_moves:
        all_solution_moves.sort()
        median_moves = all_solution_moves[len(all_solution_moves)//2]
        p25_moves = all_solution_moves[int(len(all_solution_moves)*0.25)]
        p75_moves = all_solution_moves[int(len(all_solution_moves)*0.75)]
        min_moves = min(all_solution_moves)
        max_moves = max(all_solution_moves)
        print(f"Move Distribution: min={min_moves}, 25%={p25_moves}, median={median_moves}, 75%={p75_moves}, max={max_moves}")
    
    # Log test results to file
    log_file = log_training_results(
        scramble_moves=scramble_moves,
        episode=num_tests,
        solved_episodes=solved_count,
        total_episodes=num_tests,
        success_rate=success_rate,
        recent_success_rate=success_rate,
        time_taken=0,  # We don't track time for testing
        agent_config={"test_mode": True, "checkpoint": checkpoint_path},
        recent_scrambles=recent_tests
    )
    
    return success_rate, avg_moves

def continuous_curriculum_training(max_scramble=20, min_episodes=50000, max_episodes=10000000,
                               success_threshold=95, batch_size=512, checkpoint_path=None,
                               use_pregenerated=True, checkpoint_interval=1000, recent_window=10000,
                               agent_config=None, plateau_patience=50000, required_improvement=0.5):
    """
    Train with continuous curriculum learning across multiple difficulty levels.
    Start with easy scrambles (level 1) and gradually introduce harder scrambles
    as the agent achieves success on current levels.
    
    Args:
        max_scramble: Maximum difficulty level to train up to
        min_episodes: Minimum total episodes to train for
        max_episodes: Maximum total episodes to train for
        success_threshold: Success rate threshold to introduce the next difficulty level
        batch_size: Batch size for training
        checkpoint_path: Path to a checkpoint to resume training from
        use_pregenerated: Whether to use pregenerated scrambles
        checkpoint_interval: How often to save checkpoints (in episodes)
        recent_window: Number of recent episodes to calculate success rate from
        agent_config: Dictionary of DQNAgent configuration parameters
        plateau_patience: Number of episodes to wait before advancing level if no improvement
        required_improvement: Minimum percentage point improvement required over plateau_patience episodes
    """
    print(f"\n=== Starting Continuous Curriculum Training (Max Level: {max_scramble}) ===")
    
    # Ensure scrambles exist for all levels if using pregenerated scrambles
    if use_pregenerated:
        for level in range(1, max_scramble + 1):
            if not ensure_scrambles_exist(level, use_pregenerated):
                print(f"Warning: Could not generate scrambles for level {level}. Using random scrambles for this level.")
    
    # Set default agent configuration if not provided
    default_config = {
        'learning_rate': 0.001,
        'memory_size': 150000,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.995,
        'target_update_freq': 1000,
        'double_dqn': True,
        'dueling_dqn': True,
        'prioritized_replay': True,
        'alpha': 0.6,
        'beta': 0.4,
        'beta_increment': 0.001
    }
    
    # Override defaults with provided configuration
    if agent_config:
        default_config.update(agent_config)
    
    # Initialize agent
    state_size = 6 * 9 * 6  # 6 faces, 9 stickers per face, 6 possible colors
    action_size = len(rl_agent.MOVES)
    
    agent = rl_agent.DQNAgent(
        state_size=state_size, 
        action_size=action_size,
        learning_rate=default_config['learning_rate'],
        memory_size=default_config['memory_size'],
        gamma=default_config['gamma'],
        epsilon_start=default_config['epsilon_start'],
        epsilon_min=default_config['epsilon_min'],
        epsilon_decay=default_config['epsilon_decay'],
        target_update_freq=default_config['target_update_freq'],
        double_dqn=default_config['double_dqn'],
        dueling_dqn=default_config['dueling_dqn'],
        prioritized_replay=default_config['prioritized_replay'],
        alpha=default_config['alpha'],
        beta=default_config['beta'],
        beta_increment=default_config['beta_increment']
    )
    
    # Print agent configuration
    print(f"Agent Configuration:")
    for key, value in default_config.items():
        print(f"  - {key}: {value}")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs("data/modelCheckpoints", exist_ok=True)
    
    # Single checkpoint file path
    checkpoint_file = os.path.join("data/modelCheckpoints", "cube_solver_curriculum_all")
    
    # Initialize variables
    current_episode = 0
    active_levels = [1]  # Start with level 1
    level_stats = {}
    level_plateau_counters = {}  # Track episodes since improvement for each level
    level_best_rates = {}  # Track best success rate for each level
    
    # Load checkpoint if provided
    if checkpoint_path:
        try:
            checkpoint_data = torch.load(checkpoint_path)
            if isinstance(checkpoint_data, dict) and 'model' in checkpoint_data:
                # New checkpoint format with metadata
                agent.policy_net.load_state_dict(checkpoint_data['model'])
                current_episode = checkpoint_data.get('episode', 0)
                active_levels = checkpoint_data.get('active_levels', [1])
                level_stats = checkpoint_data.get('level_stats', {})
                level_plateau_counters = checkpoint_data.get('plateau_counters', {})
                level_best_rates = checkpoint_data.get('best_rates', {})
                print(f"Loaded checkpoint from {checkpoint_path}")
                print(f"Resuming from episode {current_episode} with active levels: {active_levels}")
            else:
                # Old checkpoint format (just model weights)
                agent.policy_net.load_state_dict(checkpoint_data)
                print(f"Loaded legacy checkpoint from {checkpoint_path}")
            
            # Also load to target network
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            # If resuming training, use a lower epsilon
            if current_episode > 0:
                agent.epsilon = max(0.2, agent.epsilon_min)  # Start with moderate exploration
                print(f"Setting exploration rate (epsilon) to {agent.epsilon:.4f}")
        except Exception as e:
            print(f"Failed to load checkpoint from {checkpoint_path}, starting fresh: {e}")
    
    # Initialize statistics tracking if not loaded from checkpoint
    if not level_stats:
        level_stats = {level: {
            'episodes': 0,
            'solved': 0,
            'recent_episodes': deque(maxlen=recent_window),
            'recent_solved': deque(maxlen=recent_window),
            'rewards': deque(maxlen=100)
        } for level in range(1, max_scramble + 1)}
    else:
        # Ensure all levels have all required fields
        for level in range(1, max_scramble + 1):
            # Add level if it doesn't exist
            if level not in level_stats:
                level_stats[level] = {
                    'episodes': 0,
                    'solved': 0,
                    'recent_episodes': deque(maxlen=recent_window),
                    'recent_solved': deque(maxlen=recent_window),
                    'rewards': deque(maxlen=100)
                }
            else:
                # Ensure all required fields exist
                if 'recent_episodes' not in level_stats[level]:
                    level_stats[level]['recent_episodes'] = deque(maxlen=recent_window)
                if 'recent_solved' not in level_stats[level]:
                    level_stats[level]['recent_solved'] = deque(maxlen=recent_window)
                if 'rewards' not in level_stats[level]:
                    level_stats[level]['rewards'] = deque(maxlen=100)
                    
                # Initialize with at least one item to avoid division by zero
                if len(level_stats[level]['recent_episodes']) == 0:
                    level_stats[level]['recent_episodes'].append(0)
                if len(level_stats[level]['recent_solved']) == 0:
                    level_stats[level]['recent_solved'].append(0)
    
    # Initialize plateau detection if not loaded
    if not level_plateau_counters:
        level_plateau_counters = {level: 0 for level in range(1, max_scramble + 1)}
    
    # Initialize best rates if not loaded
    if not level_best_rates:
        level_best_rates = {level: 0 for level in range(1, max_scramble + 1)}
    
    # Initialize environments for each difficulty level
    envs = {}
    for level in range(1, max_scramble + 1):
        envs[level] = rl_agent.CubeEnvironment(
            scramble_moves=level, 
            use_pregenerated=use_pregenerated
        )
    
    # Initialize reporting variables
    training_losses = []
    recent_scrambles = []  # (level, scramble, outcome, moves, reward)
    start_time = time.time()
    
    # Main training loop - continue until all levels reach target success rate
    all_levels_complete = False
    while not all_levels_complete:
        current_episode += 1
        
        # Select a difficulty level from active levels
        # Weight selection to favor harder levels 
        weights = []
        for level in active_levels:
            # Higher weight for harder levels, with a minimum weight
            weight = max(0.1, level / sum(active_levels))
            weights.append(weight)
        
        # Normalize weights
        weights = [w/sum(weights) for w in weights]
        
        # Select level based on weights
        selected_level = random.choices(active_levels, weights=weights)[0]
        
        # Get the environment for selected level
        env = envs[selected_level]
        
        # Reset environment and get initial state
        state = env.reset()
        done = False
        moves_taken = 0
        episode_reward = 0
        
        # Get the scramble that was applied
        scramble = " ".join(env.scramble_sequence) if hasattr(env, 'scramble_sequence') else "Random scramble"
        
        # Run one episode
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            moves_taken += 1
            episode_reward += reward
            
            if done:
                solved = env.cube.is_solved()
                outcome = "Solved" if solved else "Failed"
                
                # Update statistics for this level
                level_stats[selected_level]['episodes'] += 1
                level_stats[selected_level]['recent_episodes'].append(1)
                
                if solved:
                    level_stats[selected_level]['solved'] += 1
                    level_stats[selected_level]['recent_solved'].append(1)
                else:
                    level_stats[selected_level]['recent_solved'].append(0)
                
                level_stats[selected_level]['rewards'].append(episode_reward)
                
                # Add to recent scrambles list
                recent_scrambles.append((selected_level, scramble, outcome, moves_taken, episode_reward))
                if len(recent_scrambles) > 5:
                    recent_scrambles.pop(0)
                
                break
        
        # Train the agent with experiences from memory
        loss = agent.replay(batch_size)
        if loss:
            training_losses.append(loss)
        
        # Check if we should introduce a new difficulty level or save checkpoint - every 100 episodes
        if current_episode % 100 == 0:
            # Calculate success rates and check for plateaus
            success_rates = {}
            level_data = []
            
            for level in range(1, max_scramble + 1):
                stats = level_stats[level]
                
                # Only calculate rates for levels with enough data
                if len(stats['recent_episodes']) > 500:
                    recent_success_rate = (sum(stats['recent_solved']) / len(stats['recent_episodes'])) * 100
                    success_rates[level] = recent_success_rate
                    
                    # Check for improvement
                    if level in active_levels:
                        if recent_success_rate > level_best_rates[level] + required_improvement:
                            # Reset plateau counter if we see improvement
                            level_best_rates[level] = recent_success_rate
                            level_plateau_counters[level] = 0
                        else:
                            # Increment plateau counter if no improvement
                            level_plateau_counters[level] += 100  # Because we check every 100 episodes
                
                # Store level data for display
                if stats['episodes'] > 0:
                    overall_rate = (stats['solved'] / stats['episodes']) * 100
                    recent_rate = (sum(stats['recent_solved']) / len(stats['recent_episodes'])) * 100 if stats['recent_episodes'] else 0
                    plateau_count = level_plateau_counters.get(level, 0)
                    
                    level_data.append((
                        level, 
                        level in active_levels,
                        stats['solved'], 
                        stats['episodes'],
                        overall_rate,
                        recent_rate,
                        plateau_count
                    ))
            
            # Determine if all active levels meet success criteria or have plateaued
            levels_ready_to_advance = []
            for level in active_levels:
                if level in success_rates:
                    # Check if level has reached success threshold or plateaued
                    if success_rates[level] >= success_threshold or level_plateau_counters[level] >= plateau_patience:
                        levels_ready_to_advance.append(level)
            
            # If all current active levels are ready, add the next level
            if len(levels_ready_to_advance) == len(active_levels):
                next_level = max(active_levels) + 1
                if next_level <= max_scramble:
                    active_levels.append(next_level)
                    # Reset plateau detection for the new level
                    level_plateau_counters[next_level] = 0
                    level_best_rates[next_level] = 0
                    print(f"\nUnlocked difficulty level {next_level}! (Current levels: {active_levels})")
                    
                    # Write unlock event to log
                    with open("data/curriculum_progress.log", "a") as f:
                        f.write(f"Episode {current_episode}: Unlocked level {next_level}\n")
                        
                        # Log why we advanced (threshold or plateau)
                        for level in levels_ready_to_advance:
                            if level in success_rates and success_rates[level] >= success_threshold:
                                f.write(f"  Level {level}: Reached success rate {success_rates[level]:.1f}% (threshold: {success_threshold}%)\n")
                            else:
                                f.write(f"  Level {level}: Plateaued after {level_plateau_counters[level]} episodes with best rate {level_best_rates[level]:.1f}%\n")
            
            # Calculate overall statistics
            overall_solved = sum(stats['solved'] for stats in level_stats.values())
            overall_episodes = sum(stats['episodes'] for stats in level_stats.values())
            overall_success_rate = (overall_solved / overall_episodes) * 100 if overall_episodes > 0 else 0
            
            # Average loss
            avg_loss = sum(training_losses[-100:]) / len(training_losses[-100:]) if training_losses else 0
            
            # Clear screen for cleaner display
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Display training progress
            print(f"=== Continuous Curriculum Training Progress ===")
            print(f"  - Episode: {current_episode}")
            print(f"  - Active Levels: {active_levels} (max: {max_scramble})")
            print(f"  - Success Threshold: {success_threshold}% | Plateau Patience: {plateau_patience} episodes")
            print(f"  - Batch Size: {batch_size} | Using Pregenerated: {use_pregenerated}")
            print("=" * 50)
            
            # Performance metrics
            print(f"Performance Summary:")
            print(f"✓ Overall Success Rate: {overall_success_rate:.2f}%")
            print(f"✓ Memory Size: {agent.get_memory_length()} | Epsilon: {agent.epsilon:.4f}")
            print(f"✓ Average Loss: {avg_loss:.4f}")
            print()
            
            # Level statistics
            print(f"Level Statistics:")
            print(f"{'*':1} {'Lvl':3} {'Solved':8} {'Total':8} {'Overall%':9} {'Recent%':9} {'Plateau':8}")
            print("-" * 55)
            
            for level, is_active, solved, total, overall, recent, plateau in sorted(level_data):
                active_marker = "*" if is_active else " "
                print(f"{active_marker:1} {level:3d} {solved:8d} {total:8d} {overall:8.1f}% {recent:8.1f}% {plateau:8d}")
            print()
            
            # Recent scrambles
            print(f"Last {len(recent_scrambles)} Episodes:")
            for level, scr, out, mvs, rew in recent_scrambles:
                print(f"- Level {level}: {scr} ({out} in {mvs} moves, reward: {rew:.1f})")
            
            # Save checkpoint
            if current_episode % checkpoint_interval == 0:
                # Save model with metadata
                checkpoint_data = {
                    'model': agent.policy_net.state_dict(),
                    'episode': current_episode,
                    'active_levels': active_levels,
                    'level_stats': {
                        level: {
                            'episodes': stats['episodes'],
                            'solved': stats['solved']
                        }
                        for level, stats in level_stats.items()
                    },
                    'plateau_counters': level_plateau_counters,
                    'best_rates': level_best_rates,
                    'timestamp': time.time()
                }
                
                # Save to the single checkpoint file
                torch.save(checkpoint_data, checkpoint_file + "_" + str(active_levels) + ".pt")
                print(f"\nSaved checkpoint: {checkpoint_file}")
        
        # Exit condition: Check if we have reached or exceeded max_scramble with all levels complete
        if max(active_levels) == max_scramble and current_episode >= min_episodes:
            # Check if all levels have reached the success threshold
            all_complete = True
            missing_levels = []
            
            for level in range(1, max_scramble + 1):
                stats = level_stats[level]
                if len(stats['recent_episodes']) > 500:
                    recent_success_rate = (sum(stats['recent_solved']) / len(stats['recent_episodes'])) * 100
                    if recent_success_rate < success_threshold:
                        all_complete = False
                        missing_levels.append((level, recent_success_rate))
                else:
                    all_complete = False
                    missing_levels.append((level, 0))
            
            if all_complete:
                print(f"\nAll difficulty levels 1-{max_scramble} have reached the target success rate of {success_threshold}%")
                print(f"Training completed successfully after {current_episode} episodes.")
                all_levels_complete = True
            elif current_episode >= max_episodes:
                # We've reached max_episodes but haven't completed all levels
                # Check if we've made enough progress to justify stopping
                incomplete_levels = [level for level, rate in missing_levels if rate < success_threshold * 0.8]
                
                if not incomplete_levels:
                    print(f"\nReached max episodes ({max_episodes}) with good progress on all levels.")
                    print(f"Levels not fully complete: {missing_levels}")
                    print(f"Training stopping as progress is sufficient.")
                    all_levels_complete = True
    
    # Training completed
    total_time = time.time() - start_time
    
    # Save final model (already saved to cube_solver_curriculum_all.pt during training)
    print(f"\nTraining completed after {current_episode} episodes in {total_time:.2f} seconds.")
    print(f"Final model saved to: {checkpoint_file}")
    
    # Log training results
    level_stats_summary = []
    for level in range(1, max_scramble + 1):
        stats = level_stats[level]
        if stats['episodes'] > 0:
            success_rate = (stats['solved'] / stats['episodes']) * 100
            recent_rate = (sum(stats['recent_solved']) / len(stats['recent_episodes'])) * 100 if stats['recent_episodes'] else 0
            level_stats_summary.append(f"Level {level}: {stats['solved']}/{stats['episodes']} ({success_rate:.1f}% overall, {recent_rate:.1f}% recent)")
    
    # Calculate overall statistics for the log
    overall_solved = sum(stats['solved'] for stats in level_stats.values())
    overall_episodes = sum(stats['episodes'] for stats in level_stats.values())
    overall_success_rate = (overall_solved / overall_episodes) * 100 if overall_episodes > 0 else 0
    
    log_file = log_training_results(
        scramble_moves="1-" + str(max_scramble),
        episode=current_episode,
        solved_episodes=overall_solved,
        total_episodes=overall_episodes,
        success_rate=overall_success_rate,
        recent_success_rate=overall_success_rate,  # Same as overall for the log
        time_taken=total_time,
        agent_config=default_config,
        level_stats=level_stats_summary,
        recent_scrambles=[(f"L{level}", scr, out, mvs, rew) for level, scr, out, mvs, rew in recent_scrambles],
        avg_loss=avg_loss,
        epsilon=agent.epsilon,
        memory_size=agent.get_memory_length()
    )
    
    print(f"Training results logged to: {log_file}")
    
    # Test the final model on each difficulty level
    print("\n=== Testing final model on each difficulty level ===")
    for level in range(1, max_scramble + 1):
        print(f"\nTesting on level {level}:")
        test_agent(num_tests=50, scramble_moves=level, checkpoint_path=checkpoint_file, use_pregenerated=use_pregenerated)
    
    return checkpoint_file


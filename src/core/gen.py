import kociemba as koc
import cube
import random
import json
import os
import argparse

def generate_scramble(length):
    """
    Generate a scramble whose solution length is exactly 'length'.
    
    Args:
        length: The target solution length
        
    Returns:
        tuple: (scramble_str, solution_str, kociemba_string) if a match is found, None otherwise
    """
    # Create a new cube for each attempt
    c = cube.Cube()
    
    # Generate a random scramble
    c, scramble_str = c.scramble(random.randint(length, length+3))
    
    # Get kociemba solution
    kociemba_string = c.to_kociemba_string()
    
    try:
        solution = koc.solve(kociemba_string)
        solution_length = len(solution.split())
        
        # If solution length matches target length, return the scramble and solution
        if solution_length == length:
            return scramble_str, solution, kociemba_string
        else:
            return None
    except Exception as e:
        # Skip invalid cube states
        return None

def generate_scrambles_for_level(n, target_count=50000):
    """Generate scrambles for a specific difficulty level and save to a file"""
    # Make sure scrambles directory exists
    os.makedirs("scrambles", exist_ok=True)
    
    # Output file path
    output_file = os.path.join("scrambles", f"{n}movescramble.txt")
    
    print(f"Generating {target_count} scrambles with solution length {n}...")
    print(f"Will save to {output_file}")
    
    count = 0
    attempt = 0
    
    # Open file once for writing
    with open(output_file, "w") as f:
        while count < target_count:
            result = generate_scramble(n)
            attempt += 1
            
            if result:
                scramble_str, solution, kociemba_string = result
                
                # Save scramble in a format that's easy to read from another Python function
                scramble_data = {
                    "scramble": scramble_str,
                    "solution": solution,
                    "kociemba_string": kociemba_string
                }
                
                # Write as a single line JSON object
                f.write(json.dumps(scramble_data) + "\n")
                
                # Print progress
                count += 1
                if count % 100 == 0:
                    print(f"Generated {count}/{target_count} scrambles")
            
            # Print attempt progress occasionally
            if attempt % 1000 == 0:
                print(f"Checked {attempt} scrambles, found {count} matches so far")
    
    print(f"Successfully generated {count} scrambles with solution length {n} after checking {attempt} scrambles")
    print(f"Results saved to {output_file}")
    return output_file

def generate_mixed_scrambles(levels_and_counts):
    """
    Generate scrambles for multiple difficulty levels and save to a single file.
    The scrambles will be randomly ordered in the output file.
    
    Args:
        levels_and_counts: List of tuples (level, count) specifying how many scrambles
                          to generate for each level
    
    Returns:
        str: Path to the output file
    """
    # Make sure scrambles directory exists
    os.makedirs("scrambles", exist_ok=True)
    
    # Get the highest level for the filename
    max_level = max(level for level, _ in levels_and_counts)
    
    # Output file path
    output_file = os.path.join("scrambles", f"{max_level}movescramble.txt")
    
    print(f"Generating mixed scrambles for levels: {levels_and_counts}")
    print(f"Will save to {output_file}")
    
    # First, collect all scrambles in memory
    all_scrambles = []
    
    for level, target_count in levels_and_counts:
        print(f"\nGenerating {target_count} scrambles for level {level}...")
        
        count = 0
        attempt = 0
        
        while count < target_count:
            result = generate_scramble(level)
            attempt += 1
            
            if result:
                scramble_str, solution, kociemba_string = result
                
                # Save scramble in a format that's easy to read from another Python function
                scramble_data = {
                    "scramble": scramble_str,
                    "solution": solution,
                    "kociemba_string": kociemba_string,
                    "level": level  # Add level information to the data
                }
                
                # Add to our collection
                all_scrambles.append(scramble_data)
                
                # Print progress
                count += 1
                if count % 100 == 0:
                    print(f"Generated {count}/{target_count} scrambles for level {level}")
            
            # Print attempt progress occasionally
            if attempt % 1000 == 0:
                print(f"Checked {attempt} scrambles, found {count} matches for level {level} so far")
        
        print(f"Successfully generated {count} scrambles for level {level} after checking {attempt} scrambles")
    
    # Randomize the order of all scrambles
    print(f"\nRandomizing the order of {len(all_scrambles)} scrambles...")
    random.shuffle(all_scrambles)
    
    # Write all scrambles to the file in random order
    with open(output_file, "w") as f:
        for scramble_data in all_scrambles:
            f.write(json.dumps(scramble_data) + "\n")
    
    print(f"All {len(all_scrambles)} scrambles saved to {output_file} in random order")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate scrambles with specific solution lengths')
    
    # Create a mutually exclusive group for the two modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    # Single level mode
    mode_group.add_argument('--level', type=int, 
                        help='Target solution length to generate scrambles for')
    mode_group.add_argument('--mixed', action='store_true',
                        help='Generate scrambles for multiple levels in a single file')
    
    # Common arguments
    parser.add_argument('--count', type=int, default=50000, 
                        help='Number of scrambles to generate (for single level mode)')
    
    # Mixed mode arguments
    parser.add_argument('--levels', type=str, 
                        help='Comma-separated list of level:count pairs (e.g., "6:25000,7:25000")')
    
    args = parser.parse_args()
    
    if args.mixed:
        if not args.levels:
            parser.error("--levels is required when using --mixed")
        
        # Parse the levels and counts
        try:
            levels_and_counts = []
            for pair in args.levels.split(','):
                level, count = map(int, pair.split(':'))
                levels_and_counts.append((level, count))
            
            generate_mixed_scrambles(levels_and_counts)
        except Exception as e:
            print(f"Error parsing levels and counts: {e}")
            print("Format should be 'level:count,level:count' (e.g., '6:25000,7:25000')")
    else:
        # Single level mode
        if not args.level:
            parser.error("--level is required when not using --mixed")
        
        generate_scrambles_for_level(args.level, args.count)
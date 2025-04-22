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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate scrambles with specific solution lengths')
    parser.add_argument('--level', type=int, required=True, 
                        help='Target solution length to generate scrambles for')
    parser.add_argument('--count', type=int, default=50000, 
                        help='Number of scrambles to generate')
    
    args = parser.parse_args()
    
    generate_scrambles_for_level(args.level, args.count)
import streamlit as st
import subprocess
import os
import glob
from pathlib import Path
import re
import time
import threading
import queue

st.set_page_config(page_title="RLCUBE Training Interface", layout="wide")

st.title("RLCUBE Training Interface")

def find_latest_checkpoint(pattern):
    """Find the latest checkpoint file matching the pattern"""
    files = glob.glob(pattern)
    if not files:
        return None
    
    # Try to extract numbers from filenames for better sorting
    def extract_number(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) if numbers else 0
    
    try:
        # Sort by extracted number
        return sorted(files, key=extract_number, reverse=True)[0]
    except:
        # Fall back to string sorting
        return sorted(files)[-1]

def run_command_with_live_output(cmd, use_shell=False):
    """Run a command and display live output in the Streamlit UI"""
    # Create a placeholder for the output
    output_placeholder = st.empty()
    
    # Create a queue for the output
    output_queue = queue.Queue()
    log_output = []
    
    # Function to enqueue output from the process
    def enqueue_output(out, queue):
        for line in iter(out.readline, b''):
            queue.put(line.decode())
        out.close()
    
    # Start the process
    if use_shell:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
    else:
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
    
    # Start a thread to read the process output
    t = threading.Thread(target=enqueue_output, args=(process.stdout, output_queue))
    t.daemon = True
    t.start()
    
    # Create a scrollable container for logs
    log_container = st.container()
    
    # Create a stop button
    stop_col1, stop_col2 = st.columns([1, 5])
    with stop_col1:
        stop = st.button("Stop Process")
    
    # Poll the queue for output and update the UI
    try:
        while process.poll() is None:
            # Check if user wants to stop the process
            if stop:
                process.terminate()
                st.warning("Process stopped by user")
                break
                
            # Get and display output
            while not output_queue.empty():
                line = output_queue.get_nowait()
                log_output.append(line)
                with log_container:
                    st.code("".join(log_output[-1000:]), language="bash")  # Keep last 1000 lines to avoid memory issues
            
            # Short pause to avoid consuming too many resources
            time.sleep(0.1)
            
        # Get remaining output after process completes
        while not output_queue.empty():
            line = output_queue.get_nowait()
            log_output.append(line)
        
        with log_container:
            st.code("".join(log_output[-1000:]), language="bash")
        
        if process.returncode == 0:
            st.success("Process completed successfully")
        else:
            st.error(f"Process failed with return code {process.returncode}")
            
    except Exception as e:
        st.error(f"Error: {e}")
        if process.poll() is None:
            process.terminate()

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Training", "Testing", "Solving"])

if page == "Training":
    st.header("Training Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Training")
        level = st.number_input("Training Level", min_value=1, max_value=20, value=6)
        min_rate = st.number_input("Minimum Success Rate (%)", min_value=0, max_value=100, value=90)
        
        # Check for existing model for this level
        latest_model = find_latest_checkpoint(f"data/modelCheckpoints/cube_solver_model_scramble_{level}*.pt")
        use_existing = False
        
        if latest_model:
            st.success(f"Found existing model: {os.path.basename(latest_model)}")
            use_existing = st.checkbox("Continue training from this checkpoint", value=True)
        
        if st.button("Start Training"):
            cmd = f"python src/rl_agent.py --level {level} --max_level {level} --min_rate {min_rate} --use_pregenerated --target_rate 100 --min_episodes 50000 --batch_size 128 --recent_window 10000"
            
            if use_existing and latest_model:
                cmd += f" --model \"{latest_model}\""
                st.info(f"Continuing training from: {latest_model}")
            else:
                st.info("Starting training from scratch")
                
            st.info(f"Running command: {cmd}")
            run_command_with_live_output(cmd)
    
    with col2:
        st.subheader("Continuous Curriculum Training")
        max_scramble = st.number_input("Max Scramble", min_value=1, max_value=20, value=6)
        min_episodes = st.number_input("Min Episodes", min_value=1000, value=1000000)
        max_episodes = st.number_input("Max Episodes", min_value=1000, value=1000000)
        success_threshold = st.number_input("Success Threshold (%)", min_value=0, max_value=100, value=90)
        batch_size = st.number_input("Batch Size", min_value=32, value=512)
        plateau_patience = st.number_input("Plateau Patience", min_value=1, value=1000000)
        
        # Check for existing curriculum checkpoint
        latest_curriculum = find_latest_checkpoint("data/modelCheckpoints/cube_solver_curriculum*.pt")
        use_curriculum = False
        checkpoint_path = ""
        
        if latest_curriculum:
            st.success(f"Found existing curriculum checkpoint: {os.path.basename(latest_curriculum)}")
            use_curriculum = st.checkbox("Continue training from this curriculum checkpoint", value=True)
            if use_curriculum:
                checkpoint_path = latest_curriculum
        
        if not use_curriculum:
            checkpoint_path = st.text_input("Checkpoint Path (leave empty to start from scratch)", value="")

        if st.button("Start Curriculum Training"):
            if checkpoint_path:
                cmd = f"python -c \"import sys; sys.path.append('src'); import helper, rl_agent; helper.continuous_curriculum_training(max_scramble={max_scramble}, min_episodes={min_episodes}, max_episodes={max_episodes}, success_threshold={success_threshold}, batch_size={batch_size}, checkpoint_path='{checkpoint_path}', use_pregenerated=True, plateau_patience={plateau_patience})\""
                st.info(f"Continuing curriculum training from: {checkpoint_path}")
            else:
                cmd = f"python -c \"import sys; sys.path.append('src'); import helper, rl_agent; helper.continuous_curriculum_training(max_scramble={max_scramble}, min_episodes={min_episodes}, max_episodes={max_episodes}, success_threshold={success_threshold}, batch_size={batch_size}, use_pregenerated=True, plateau_patience={plateau_patience})\""
                st.info("Starting curriculum training from scratch")
            
            run_command_with_live_output(cmd, use_shell=True)

elif page == "Testing":
    st.header("Testing Options")
    
    scramble = st.number_input("Scramble Moves", min_value=1, max_value=20, value=6)
    num_tests = st.number_input("Number of Tests", min_value=1, value=100)
    
    # Get available model checkpoints
    model_dir = Path("data/modelCheckpoints")
    if model_dir.exists():
        model_files = list(model_dir.glob("**/*.pt"))
        model_paths = [str(p.relative_to(model_dir)) for p in model_files]
        selected_model = st.selectbox("Select Model", model_paths)
    else:
        st.warning("No model checkpoints found!")
        selected_model = None
    
    if st.button("Run Tests") and selected_model:
        cmd = f"python src/test_rl_agent.py --scramble {scramble} --tests {num_tests} --model \"data/modelCheckpoints/{selected_model}\" --use_pregenerated"
        st.info(f"Running tests...")
        run_command_with_live_output(cmd)

else:  # Solving page
    st.header("Cube Solving")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Interactive Solving")
        
        # Get available model checkpoints for solving
        model_dir = Path("data/modelCheckpoints")
        if model_dir.exists():
            model_files = list(model_dir.glob("**/*.pt"))
            model_paths = [str(p.relative_to(model_dir)) for p in model_files]
            selected_solve_model = st.selectbox("Select Model for Interactive Solving", model_paths)
        else:
            st.warning("No model checkpoints found!")
            selected_solve_model = None
        
        if st.button("Start Interactive Solver") and selected_solve_model:
            cmd = f"python src/advanced_solver.py --interactive --model \"data/modelCheckpoints/{selected_solve_model}\""
            st.info("Starting interactive solver...")
            run_command_with_live_output(cmd)
    
    with col2:
        st.subheader("Benchmark Solving")
        scramble_moves = st.number_input("Scramble Moves", min_value=1, max_value=20, value=6)
        num_tests = st.number_input("Number of Tests", min_value=1, value=100)
        
        # Get available model checkpoints for benchmark
        if model_dir.exists():
            model_files = list(model_dir.glob("**/*.pt"))
            model_paths = [str(p.relative_to(model_dir)) for p in model_files]
            selected_benchmark_model = st.selectbox("Select Model for Benchmark", model_paths)
        else:
            st.warning("No model checkpoints found!")
            selected_benchmark_model = None
        
        if st.button("Run Benchmark") and selected_benchmark_model:
            cmd = f"python src/advanced_solver.py --benchmark --scramble_moves {scramble_moves} --tests {num_tests} --use_pregenerated --model \"data/modelCheckpoints/{selected_benchmark_model}\""
            st.info("Running benchmark...")
            run_command_with_live_output(cmd) 
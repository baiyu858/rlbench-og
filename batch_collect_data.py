#!/usr/bin/env python3
"""
Batch data collection script for RLBench tasks.

This script sequentially collects demonstration data for multiple tasks,
with support for checkpoint resume and logging.

Usage:
    python batch_collect_data.py
"""

import os
import subprocess
import sys
from datetime import datetime

# Configuration
BASE_TASKS = [
    "solve_puzzle",
    "block_pyramid",
    "close_drawer",
    "straighten_rope",
    "take_plate_off_colored_dish_rack",
    "toilet_seat_down",
    "take_usb_out_of_computer",
    "water_plants",
    "scoop_with_spatula",
    "basketball_in_hoop",
]

TRAIN_PATH = "/home/hcp/rlbench_og/data/train"
TEST_PATH = "/home/hcp/rlbench_og/data/test"
PROCESSES = 1
LOG_FILE = "batch_collection.log"

# Task list: (task_name, save_path, episodes, description)
TASKS_TO_COLLECT = []

# 1. Occlusion tasks for training (50 episodes each) - COLLECT FIRST
for task in BASE_TASKS:
    TASKS_TO_COLLECT.append((
        f"{task}_occlusion",
        TRAIN_PATH,
        50,
        f"Training data (occlusion) for {task}"
    ))

# 2. Occlusion tasks for testing (25 episodes each) - COLLECT SECOND
for task in BASE_TASKS:
    TASKS_TO_COLLECT.append((
        f"{task}_occlusion",
        TEST_PATH,
        25,
        f"Test data (occlusion) for {task}"
    ))

# 3. Normal tasks for training (50 episodes each) - COLLECT LAST
for task in BASE_TASKS:
    TASKS_TO_COLLECT.append((
        task,
        TRAIN_PATH,
        50,
        f"Training data (normal) for {task}"
    ))


def log_message(message, also_print=True):
    """Log a message to the log file and optionally print to console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    
    with open(LOG_FILE, "a") as f:
        f.write(log_line)
    
    if also_print:
        print(log_line, end="")


def check_task_completed(task_name, save_path, expected_episodes):
    """
    Check if a task has already been collected by counting episodes.
    
    Returns:
        bool: True if task is complete, False otherwise
    """
    task_path = os.path.join(save_path, task_name, "all_variations", "episodes")
    
    if not os.path.exists(task_path):
        return False
    
    # Count episode folders
    episode_folders = [
        d for d in os.listdir(task_path)
        if d.startswith("episode") and os.path.isdir(os.path.join(task_path, d))
    ]
    
    actual_episodes = len(episode_folders)
    
    if actual_episodes >= expected_episodes:
        log_message(f"  Task already completed: {actual_episodes}/{expected_episodes} episodes found")
        return True
    else:
        log_message(f"  Task partially completed: {actual_episodes}/{expected_episodes} episodes found, will collect remaining")
        return False


def run_collection(task_name, save_path, episodes):
    """
    Run data collection for a single task.
    
    Returns:
        bool: True if successful, False if failed
    """
    cmd = [
        "python",
        "tools/dataset_generator.py",
        "--tasks", task_name,
        "--processes", str(PROCESSES),
        "--episodes_per_task", str(episodes),
        "--save_path", save_path
    ]
    
    log_message(f"  Running command: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output to both console and log file
        for line in iter(process.stdout.readline, ''):
            if line:
                log_message(f"    {line.rstrip()}", also_print=True)
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            log_message(f"  ✅ Task completed successfully")
            return True
        else:
            log_message(f"  ❌ Task failed with return code {return_code}")
            return False
            
    except Exception as e:
        log_message(f"  ❌ Error running task: {str(e)}")
        return False


def main():
    """Main entry point."""
    
    log_message("=" * 80)
    log_message("Starting batch data collection")
    log_message("=" * 80)
    log_message(f"Total tasks to collect: {len(TASKS_TO_COLLECT)}")
    log_message(f"Processes per task: {PROCESSES}")
    log_message("")
    
    # Create directories if they don't exist
    os.makedirs(TRAIN_PATH, exist_ok=True)
    os.makedirs(TEST_PATH, exist_ok=True)
    
    completed_tasks = 0
    skipped_tasks = 0
    failed_tasks = 0
    
    for idx, (task_name, save_path, episodes, description) in enumerate(TASKS_TO_COLLECT, 1):
        log_message("")
        log_message("-" * 80)
        log_message(f"Task {idx}/{len(TASKS_TO_COLLECT)}: {description}")
        log_message(f"  Task name: {task_name}")
        log_message(f"  Save path: {save_path}")
        log_message(f"  Episodes: {episodes}")
        log_message("-" * 80)
        
        # Check if task is already completed
        if check_task_completed(task_name, save_path, episodes):
            log_message(f"  ⏭️  Skipping already completed task")
            skipped_tasks += 1
            continue
        
        # Run collection
        success = run_collection(task_name, save_path, episodes)
        
        if success:
            completed_tasks += 1
        else:
            failed_tasks += 1
            log_message(f"  ⚠️  Task failed, continuing to next task...")
    
    # Final summary
    log_message("")
    log_message("=" * 80)
    log_message("Batch collection completed!")
    log_message("=" * 80)
    log_message(f"Total tasks: {len(TASKS_TO_COLLECT)}")
    log_message(f"Completed: {completed_tasks}")
    log_message(f"Skipped (already done): {skipped_tasks}")
    log_message(f"Failed: {failed_tasks}")
    log_message("=" * 80)
    
    if failed_tasks > 0:
        log_message("⚠️  Some tasks failed. Check the log for details.")
        return 1
    else:
        log_message("✅ All tasks processed successfully!")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log_message("")
        log_message("=" * 80)
        log_message("⚠️  Collection interrupted by user (Ctrl+C)")
        log_message("=" * 80)
        log_message("You can resume by running this script again.")
        log_message("Already completed tasks will be skipped.")
        sys.exit(130)
    except Exception as e:
        log_message("")
        log_message("=" * 80)
        log_message(f"❌ Unexpected error: {str(e)}")
        log_message("=" * 80)
        import traceback
        log_message(traceback.format_exc())
        sys.exit(1)

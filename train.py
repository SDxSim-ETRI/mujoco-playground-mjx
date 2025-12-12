#!/usr/bin/env python3
"""
MuJoCo Playground PPO Training Script

Simple wrapper to run training from the project root directory.
Usage:
    python train.py --env_name HumanoidStand --num_timesteps 50000000 --num_envs 2048 --use_tb --num_videos 5
"""

import os
import sys
import subprocess

def main():
    # Change to mujoco_playground directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mujoco_playground_dir = os.path.join(script_dir, "mujoco_playground")
    
    # Check if mujoco_playground directory exists
    if not os.path.exists(mujoco_playground_dir):
        print(f"Error: mujoco_playground directory not found at {mujoco_playground_dir}")
        sys.exit(1)
    
    # Path to the actual training script
    train_script = os.path.join(mujoco_playground_dir, "learning", "train_jax_ppo.py")
    
    if not os.path.exists(train_script):
        print(f"Error: Training script not found at {train_script}")
        sys.exit(1)
    
    # Build the command with all arguments passed to this script
    cmd = [sys.executable, train_script] + sys.argv[1:]
    
    # Change to mujoco_playground directory and run
    os.chdir(mujoco_playground_dir)
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    print("-" * 80)
    
    # Execute the training script
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Complete FreeTimeGS pipeline: Video -> 4D Gaussian Splatting -> Web Viewer
"""

import subprocess
import os
import sys
from pathlib import Path
import argparse

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def find_video_file(videos_dir="videos"):
    """Find the first video file in the videos directory"""
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    videos_path = Path(videos_dir)
    
    if not videos_path.exists():
        print(f"Videos directory '{videos_dir}' not found")
        return None
    
    for ext in video_exts:
        for video_file in videos_path.glob(f"*{ext}"):
            return str(video_file)
    
    print(f"No video files found in '{videos_dir}'")
    return None


def main():
    parser = argparse.ArgumentParser(description="Run complete FreeTimeGS pipeline")
    parser.add_argument("--video", help="Path to video file (auto-detect if not provided)")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, just run viewer")
    parser.add_argument("--iterations", type=int, default=5000, help="Training iterations (reduced for demo)")
    
    args = parser.parse_args()
    
    
    # Find video file
    video_path = args.video if args.video else find_video_file()
    if not video_path and not args.skip_training:
        print("No video file specified or found. Please:")
        print("1. Add a video file to the 'videos' directory, or")
        print("2. Specify a video path with --video, or")
        print("3. Use --skip-training to just run the viewer")
        sys.exit(1)

   
    
    if not args.skip_training:
        # Step 1: Process video
        print(f"\nProcessing video: {video_path}")
        if not run_command([
            "python", "process_video.py", 
            video_path,
            "--output_dir", "output",
            "--fps", "10",
            "--max_frames", "300"
        ], "Processing video and extracting frames"):
            print("Note: COLMAP might not be installed. Continuing with dummy camera data...")
        
        # Step 2: Train FreeTimeGS
        print("\nTraining FreeTimeGS model...")
        if not run_command([
            "python", "freetimegs_trainer.py",
            "output",
            "--output_dir", "trained_model",
            "--iterations", str(args.iterations)
        ], "Training 4D Gaussian Splatting model"):
            
            # Create dummy data for demo
            raise Exception("Training failed. You may need a GPU for optimal performance.")
            # create_dummy_model()
    
    # Step 3: Launch web viewer
    print("\nLaunching web viewer...")
    print("="*50)
    print("FreeTimeGS 4D Gaussian Splatting Viewer")
    print("Open http://localhost:8000 in your browser")
    print("Press Ctrl+C to stop the server")
    print("="*50)
    
    try:
        subprocess.run(["python", "web_viewer.py"], check=True)
    except KeyboardInterrupt:
        print("\nViewer stopped.")

def create_dummy_model():
    """Create dummy model data for demo purposes"""
    import json
    import numpy as np
    
    print("Creating dummy model data for demo...")
    
    os.makedirs("trained_model", exist_ok=True)
    
    # Create a simple 3D scene with temporal dynamics
    num_gaussians = 1000
    
    # Create a sphere of points
    phi = np.random.uniform(0, 2*np.pi, num_gaussians)
    costheta = np.random.uniform(-1, 1, num_gaussians)
    theta = np.arccos(costheta)
    
    r = np.random.uniform(0.5, 1.5, num_gaussians)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    positions = np.column_stack([x, y, z])
    
    # Random colors
    colors = np.random.rand(num_gaussians, 3)
    
    # Random opacities
    opacities = np.random.rand(num_gaussians, 1) * 0.8 + 0.2
    
    # Simple temporal dynamics - rotation
    velocities = np.column_stack([
        -y * 0.5,  # Rotate around Z axis
        x * 0.5,
        np.random.randn(num_gaussians) * 0.1
    ])
    
    temporal_widths = np.ones((num_gaussians, 1)) * 0.3
    
    gaussians_data = {
        'positions': positions.tolist(),
        'colors': colors.tolist(),
        'opacities': opacities.tolist(),
        'velocities': velocities.tolist(),
        'temporal_widths': temporal_widths.tolist(),
        'num_gaussians': num_gaussians,
        'max_time': 1.0,
    }
    
    with open("trained_model/gaussians.json", 'w') as f:
        json.dump(gaussians_data, f)
    
    print("✓ Dummy model created successfully")

if __name__ == "__main__":
    main() 
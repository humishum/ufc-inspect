#!/usr/bin/env python3
"""
Video processing pipeline for FreeTimeGS 4D Gaussian Splatting
Extracts frames, runs COLMAP for camera pose estimation
"""

import os
import cv2
import numpy as np
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_frames(video_path: str, output_dir: str, fps: int = 10, max_frames: int = 300):
    """Extract frames from video at specified fps"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    extracted_count = 0
    
    print(f"Extracting frames at {fps} FPS (every {frame_interval} frames)")
    
    while cap.isOpened() and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames to {output_dir}")
    return extracted_count

def run_colmap_sfm(images_dir: str, output_dir: str):
    """Run COLMAP Structure from Motion"""
    os.makedirs(output_dir, exist_ok=True)
    
    # COLMAP feature extraction
    print("Running COLMAP feature extraction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", os.path.join(output_dir, "database.db"),
        "--image_path", images_dir,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "PINHOLE"
    ], check=True)
    
    # COLMAP matching
    print("Running COLMAP feature matching...")
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", os.path.join(output_dir, "database.db")
    ], check=True)
    
    # COLMAP reconstruction
    print("Running COLMAP reconstruction...")
    reconstruction_dir = os.path.join(output_dir, "sparse")
    os.makedirs(reconstruction_dir, exist_ok=True)
    
    subprocess.run([
        "colmap", "mapper",
        "--database_path", os.path.join(output_dir, "database.db"),
        "--image_path", images_dir,
        "--output_path", reconstruction_dir
    ], check=True)
    
    print(f"COLMAP reconstruction completed in {reconstruction_dir}")

def main():
    parser = argparse.ArgumentParser(description="Process video for FreeTimeGS")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS for frame extraction")
    parser.add_argument("--max_frames", type=int, default=300, help="Maximum frames to extract")
    
    args = parser.parse_args()
    
    # Create output structure
    base_output = Path(args.output_dir)
    images_dir = base_output / "images"
    colmap_dir = base_output / "colmap"
    
    # Extract frames
    extract_frames(args.video_path, str(images_dir), args.fps, args.max_frames)
    
    # Run COLMAP (if available)
    try:
        run_colmap_sfm(str(images_dir), str(colmap_dir))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"COLMAP not available or failed: {e}")
        print("Please install COLMAP or provide camera poses manually")

if __name__ == "__main__":
    main() 
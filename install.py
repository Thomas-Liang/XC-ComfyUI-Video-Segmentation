#!/usr/bin/env python3
"""
Installation script for ComfyUI Video Segmentation Node
This script helps users install all required dependencies automatically.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path

def run_command(cmd):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ Command succeeded: {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        return False

def install_requirements():
    """Install Python requirements."""
    print("Installing Python dependencies...")
    req_file = Path(__file__).parent / "requirements.txt"
    
    if req_file.exists():
        return run_command(f"{sys.executable} -m pip install -r {req_file}")
    else:
        print("✗ requirements.txt not found!")
        return False

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    print("Checking FFmpeg installation...")
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✓ FFmpeg is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ FFmpeg not found!")
        print("Please install FFmpeg:")
        print("  Windows: https://ffmpeg.org/download.html")
        print("  Linux: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        return False

def install_transnetv2():
    """Try to install TransNetV2."""
    print("Installing TransNetV2...")
    return run_command(f"{sys.executable} -m pip install transnetv2")

def download_model_weights():
    """Download TransNetV2 model weights if needed."""
    print("Checking TransNetV2 model weights...")
    
    weights_dir = Path(__file__).parent / "transnetv2-weights"
    if weights_dir.exists():
        print("✓ Model weights already exist")
        return True
    
    print("Model weights not found. Please download them manually from:")
    print("https://github.com/soCzech/TransNetV2/tree/master/inference/transnetv2-weights")
    print(f"And place them in: {weights_dir}")
    return False

def main():
    """Main installation routine."""
    print("ComfyUI Video Segmentation Node - Installation Script")
    print("=" * 55)
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Install Python requirements
    if install_requirements():
        success_count += 1
    
    # Step 2: Check FFmpeg
    if check_ffmpeg():
        success_count += 1
    
    # Step 3: Install TransNetV2
    if install_transnetv2():
        success_count += 1
    
    # Step 4: Check model weights
    if download_model_weights():
        success_count += 1
    
    print("\n" + "=" * 55)
    print(f"Installation completed: {success_count}/{total_steps} steps successful")
    
    if success_count == total_steps:
        print("✓ All dependencies installed successfully!")
        print("You can now restart ComfyUI and use the Video Segmentation node.")
    else:
        print("⚠ Some steps failed. Please check the errors above.")
        print("The node may still work with reduced functionality.")
    
    print("\nManual steps you may need to complete:")
    print("1. Download TransNetV2 weights if not done automatically")
    print("2. Install FFmpeg if not already available")
    print("3. Restart ComfyUI")

if __name__ == "__main__":
    main()
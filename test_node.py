"""
Test script for the Video Segmentation node.
This script tests the basic structure without requiring ComfyUI dependencies.
"""

import os
import sys

def test_node_structure():
    """Test basic node structure and configuration."""
    print("Testing Video Segmentation Node Structure...")
    
    # Test file existence
    required_files = ['__init__.py', 'nodes.py', 'README.md']
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file} exists")
        else:
            print(f"[ERROR] {file} missing")
            return False
    
    # Test basic syntax by reading the files
    try:
        with open('nodes.py', 'r') as f:
            content = f.read()
            
        # Check for required class and methods
        if 'class VideoSegmentation:' in content:
            print("[OK] VideoSegmentation class found")
        else:
            print("[ERROR] VideoSegmentation class not found")
            return False
            
        if 'INPUT_TYPES' in content:
            print("[OK] INPUT_TYPES method found")
        else:
            print("[ERROR] INPUT_TYPES method not found")
            return False
            
        if 'segment_video' in content:
            print("[OK] segment_video method found")
        else:
            print("[ERROR] segment_video method not found")
            return False
            
        if 'NODE_CLASS_MAPPINGS' in content:
            print("[OK] NODE_CLASS_MAPPINGS found")
        else:
            print("[ERROR] NODE_CLASS_MAPPINGS not found")
            return False
            
        print("[OK] All basic structure tests passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error reading nodes.py: {e}")
        return False

def test_input_types_structure():
    """Test INPUT_TYPES structure without importing."""
    print("\nTesting INPUT_TYPES structure...")
    
    with open('nodes.py', 'r') as f:
        content = f.read()
    
    # Check for required inputs
    required_inputs = ['video', 'output_dir', 'min_scene_length', 'threshold']
    for input_name in required_inputs:
        if f'"{input_name}"' in content:
            print(f"[OK] {input_name} input found")
        else:
            print(f"[ERROR] {input_name} input missing")
            return False
    
    print("[OK] All required inputs found")
    return True

def test_return_types():
    """Test return types configuration."""
    print("\nTesting return types...")
    
    with open('nodes.py', 'r') as f:
        content = f.read()
    
    if 'RETURN_TYPES = ("STRING", "INT", "LIST")' in content:
        print("[OK] Correct RETURN_TYPES found")
    else:
        print("[ERROR] RETURN_TYPES incorrect or missing")
        return False
        
    if 'RETURN_NAMES = ("output_directory", "num_segments", "segment_paths")' in content:
        print("[OK] Correct RETURN_NAMES found")
    else:
        print("[ERROR] RETURN_NAMES incorrect or missing")
        return False
    
    return True

if __name__ == "__main__":
    print("ComfyUI Video Segmentation Node Test")
    print("=" * 40)
    
    success = True
    success &= test_node_structure()
    success &= test_input_types_structure()  
    success &= test_return_types()
    
    if success:
        print("\n[SUCCESS] All tests passed! The node should work correctly in ComfyUI.")
    else:
        print("\n[FAILED] Some tests failed. Please check the implementation.")
    
    print("\nNote: This test only checks basic structure.")
    print("Full functionality testing requires ComfyUI environment with torch, etc.")
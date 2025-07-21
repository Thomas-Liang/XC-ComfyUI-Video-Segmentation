import os
import sys
import subprocess
import logging
import tempfile
import shutil
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image
import folder_paths

# Set up logging
logger = logging.getLogger(__name__)

class VideoSegmentation:
    """
    A ComfyUI node for video scene segmentation using TransNetV2.
    
    This node takes a video input and segments it into scenes using the 
    TransNetV2 project (https://github.com/soCzech/TransNetV2).
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input types for the Video Segmentation node.
        
        Returns:
            dict: Configuration for input fields
        """
        return {
            "required": {
                "video": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Path to input video file"
                }),
                "output_dir": ("STRING", {
                    "multiline": False, 
                    "default": "",
                    "placeholder": "Directory to save segmented videos"
                }),
                "min_scene_length": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 300,
                    "step": 1,
                    "display": "number"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "number"
                })
            },
        }

    RETURN_TYPES = ("STRING", "INT", "LIST")
    RETURN_NAMES = ("output_directory", "num_segments", "segment_paths")
    
    FUNCTION = "segment_video"
    
    OUTPUT_NODE = True
    
    CATEGORY = "Video/Segmentation"

    def segment_video(self, video, output_dir, min_scene_length, threshold):
        """
        Segment the input video into scenes using TransNetV2.
        
        Args:
            video (str): Path to the input video file
            output_dir (str): Directory to save segmented videos
            min_scene_length (int): Minimum scene length in frames
            threshold (float): Scene detection threshold
            
        Returns:
            Tuple: (output_directory, num_segments, segment_paths)
        """
        try:
            # Validate input video path
            if not video or not os.path.isfile(video):
                logger.error(f"Video file not found: {video}")
                return ("", 0, [])
            
            # Create output directory if it doesn't exist
            if not output_dir:
                output_dir = os.path.join(folder_paths.get_output_directory(), "video_segments")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert to absolute paths
            video_path = os.path.abspath(video)
            output_path = os.path.abspath(output_dir)
            
            logger.info(f"Starting video segmentation for: {video_path}")
            logger.info(f"Output directory: {output_path}")
            
            # Run TransNetV2 segmentation
            success = self._run_transnetv2(video_path, output_path, min_scene_length, threshold)
            
            if not success:
                logger.error("TransNetV2 segmentation failed")
                return (output_path, 0, [])
            
            # Get list of generated segment files
            segment_paths = self._get_segment_files(output_path)
            num_segments = len(segment_paths)
            
            logger.info(f"Successfully created {num_segments} video segments")
            
            return (output_path, num_segments, segment_paths)
            
        except Exception as e:
            logger.error(f"Error in video segmentation: {str(e)}")
            return ("", 0, [])

    def _run_transnetv2(self, video_path: str, output_dir: str, min_scene_length: int, threshold: float) -> bool:
        """
        Run TransNetV2 scene segmentation on the video.
        
        This method is adapted from the run_transnetv2 function in task_utils.py
        
        Args:
            video_path (str): Path to the video file
            output_dir (str): Directory to save segmented videos
            min_scene_length (int): Minimum scene length in frames
            threshold (float): Scene detection threshold
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Running TransNetV2 scene segmentation on {video_path}")
            
            # Check if video file exists
            if not os.path.isfile(video_path):
                logger.error(f"Video file not found: {video_path}")
                return False
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get Python executable
            python_executable = sys.executable
            
            # Try to find TransNetV2 script or installation
            transnetv2_script = None
            transnetv2_installed = False
            
            # First, check if TransNetV2 is installed as a Python package
            try:
                import transnetv2
                transnetv2_installed = True
                logger.info("Found TransNetV2 Python package")
            except ImportError:
                logger.info("TransNetV2 Python package not found, looking for local installation")
            
            # Look for transnetv2 in common locations
            possible_locations = [
                # Try relative to the project root from task_utils.py pattern  
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "transnetv2", "main.py"),
                # Try in current directory
                os.path.join(os.path.dirname(__file__), "transnetv2", "main.py"),
                # Try in node directory
                os.path.join(os.path.dirname(__file__), "transnetv2-weights", "inference.py"),
            ]
            
            for location in possible_locations:
                if os.path.isfile(location):
                    transnetv2_script = location
                    logger.info(f"Found TransNetV2 script at: {location}")
                    break
            
            # Construct the command based on what we found
            cmd = None
            
            if transnetv2_installed:
                # Use the installed Python package
                cmd = [
                    python_executable, "-m", "transnetv2.inference",
                    "--video-path", video_path,
                    "--output-dir", output_dir,
                    "--threshold", str(threshold)
                ]
            elif transnetv2_script:
                # Use local script
                cmd = [
                    python_executable,
                    transnetv2_script,
                    "--video_path", video_path,
                    "--output_path", output_dir,
                    "--threshold", str(threshold)
                ]
            else:
                # No TransNetV2 found, skip to fallback
                logger.warning("TransNetV2 not found, using fallback scene detection")
                return self._fallback_scene_detection(video_path, output_dir, min_scene_length)
            
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            # Execute the command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"TransNetV2 failed with return code {process.returncode}")
                logger.error(f"Error output: {stderr}")
                logger.error(f"Standard output: {stdout}")
                
                # Try alternative approach using OpenCV-based scene detection as fallback
                logger.info("Falling back to OpenCV-based scene detection")
                return self._fallback_scene_detection(video_path, output_dir, min_scene_length)
            
            logger.info(f"TransNetV2 completed successfully for {video_path}")
            logger.info(f"Output: {stdout}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error running TransNetV2: {str(e)}")
            # Try fallback method
            return self._fallback_scene_detection(video_path, output_dir, min_scene_length)

    def _fallback_scene_detection(self, video_path: str, output_dir: str, min_scene_length: int) -> bool:
        """
        Fallback scene detection using OpenCV when TransNetV2 is not available.
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Output directory for segments
            min_scene_length (int): Minimum scene length in frames
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import cv2
            
            logger.info("Using OpenCV-based fallback scene detection")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Simple scene detection based on frame differences
            scene_boundaries = self._detect_scenes_opencv(cap, min_scene_length)
            cap.release()
            
            # Create video segments
            return self._create_segments(video_path, scene_boundaries, output_dir, fps)
            
        except ImportError:
            logger.error("OpenCV not available for fallback scene detection")
            return False
        except Exception as e:
            logger.error(f"Error in fallback scene detection: {str(e)}")
            return False

    def _detect_scenes_opencv(self, cap, min_scene_length: int, threshold: float = 0.3) -> List[Tuple[int, int]]:
        """
        Detect scene boundaries using OpenCV frame difference analysis.
        
        Args:
            cap: OpenCV VideoCapture object
            min_scene_length (int): Minimum scene length in frames
            threshold (float): Threshold for scene change detection
            
        Returns:
            List[Tuple[int, int]]: List of (start_frame, end_frame) tuples
        """
        import cv2
        
        prev_frame = None
        frame_diffs = []
        frame_index = 0
        
        # Calculate frame differences
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate absolute difference
                diff = cv2.absdiff(gray, prev_frame)
                diff_sum = np.sum(diff)
                diff_norm = diff_sum / (diff.shape[0] * diff.shape[1] * 255)
                frame_diffs.append(diff_norm)
            
            prev_frame = gray
            frame_index += 1
        
        if not frame_diffs:
            return [(0, frame_index - 1)]
        
        # Find scene boundaries
        mean_diff = np.mean(frame_diffs)
        std_diff = np.std(frame_diffs)
        scene_threshold = mean_diff + 2 * std_diff
        
        boundaries = []
        for i, diff in enumerate(frame_diffs):
            if diff > scene_threshold:
                boundaries.append(i + 1)
        
        # Convert to scenes
        scenes = []
        if not boundaries:
            scenes.append((0, frame_index - 1))
        else:
            scenes.append((0, boundaries[0]))
            for i in range(len(boundaries) - 1):
                scenes.append((boundaries[i], boundaries[i + 1]))
            scenes.append((boundaries[-1], frame_index - 1))
        
        # Filter short scenes
        scenes = [(start, end) for start, end in scenes if end - start >= min_scene_length]
        
        return scenes

    def _create_segments(self, video_path: str, scenes: List[Tuple[int, int]], output_dir: str, fps: float) -> bool:
        """
        Create video segments using ffmpeg.
        
        Args:
            video_path (str): Input video path
            scenes (List[Tuple[int, int]]): Scene boundaries
            output_dir (str): Output directory
            fps (float): Video frame rate
            
        Returns:
            bool: True if successful
        """
        try:
            for i, (start_frame, end_frame) in enumerate(scenes):
                start_time = start_frame / fps
                duration = (end_frame - start_frame) / fps
                
                output_file = os.path.join(output_dir, f"segment_{i:03d}.mp4")
                
                # Use ffmpeg to extract segment
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-ss", str(start_time),
                    "-t", str(duration),
                    "-c", "copy",
                    output_file
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"Failed to create segment {i}: {result.stderr}")
                    continue
                
                logger.info(f"Created segment {i}: {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating segments: {str(e)}")
            return False

    def _get_segment_files(self, output_dir: str) -> List[str]:
        """
        Get list of generated segment files.
        
        Args:
            output_dir (str): Output directory
            
        Returns:
            List[str]: List of segment file paths
        """
        segment_files = []
        
        if not os.path.exists(output_dir):
            return segment_files
        
        # Look for video files in the output directory
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        for filename in os.listdir(output_dir):
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                segment_files.append(os.path.join(output_dir, filename))
        
        # Sort files naturally
        segment_files.sort()
        
        return segment_files


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VideoSegmentation": VideoSegmentation
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSegmentation": "Video Segmentation"
}
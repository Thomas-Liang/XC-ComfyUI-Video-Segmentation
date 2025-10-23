import os
import uuid
import folder_paths
import numpy as np
import logging
import subprocess
from typing import List, Tuple
from pathlib import Path

import torch
from PIL import Image

# Set up logging
logger = logging.getLogger(__name__)

# Try to import VIDEO input type from ComfyUI API
try:
    from comfy_api.input import VideoInput
except ImportError:
    VideoInput = None
    logger.warning("ComfyUI API not available, using fallback video handling")

# Model directory setup - same as Qwen2.5-VL
model_directory = os.path.join(folder_paths.models_dir, "VLM")
os.makedirs(model_directory, exist_ok=True)


class DownloadAndLoadTransNetModel:
    """
    A ComfyUI node for downloading and loading TransNetV2 models.
    Automatically downloads from Hugging Face (MiaoshouAI/transnetv2-pytorch-weights) if not found locally.
    Following the Qwen2.5-VL structure for consistency.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    [
                        "transnetv2-pytorch-weights",
                    ],
                    {"default": "transnetv2-weights"},
                ),
                "device": (
                    ["auto", "cpu", "cuda"],
                    {"default": "auto"},
                ),
            },
        }

    RETURN_TYPES = ("TRANSNET_MODEL",)
    RETURN_NAMES = ("TransNet_model",)
    FUNCTION = "DownloadAndLoadTransNetModel"
    CATEGORY = "MiaoshouAI Video Segmentation"

    def DownloadAndLoadTransNetModel(self, model, device):
        TransNet_model = {"model": "", "model_path": ""}
        model_name = model
        model_path = os.path.join(model_directory, model_name)

        if not os.path.exists(model_path):
            print(f"Downloading TransNetV2 model to: {model_path}")
            self._download_model(model_name, model_path)

        # Load TransNetV2 model
        try:
            import transnetv2_pytorch as transnetv2
            
            # Initialize the model
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Path to pre-converted PyTorch weights
            pytorch_weights_path = os.path.join(model_path, "transnetv2-pytorch-weights.pth")
            
            # Load the model with pre-converted PyTorch weights
            model_instance = transnetv2.TransNetV2()
            
            # Load the converted weights
            if os.path.exists(pytorch_weights_path):
                model_instance.load_state_dict(torch.load(pytorch_weights_path, map_location=device))
                logger.info(f"Loaded pre-converted PyTorch weights from {pytorch_weights_path}")
            else:
                logger.error(f"Pre-converted PyTorch weights not found at {pytorch_weights_path}")
                logger.error("Please run the weight conversion script first")
                raise FileNotFoundError(f"PyTorch weights not found: {pytorch_weights_path}")
            
            # Move model to device
            model_instance = model_instance.to(device)
            model_instance.eval()
            
            TransNet_model["model"] = model_instance
            TransNet_model["model_path"] = model_path
            TransNet_model["device"] = device
            
            logger.info(f"TransNetV2 model loaded successfully from {model_path}")
            
        except ImportError:
            logger.error("TransNetV2 package not found. Please install: pip install transnetv2-pytorch")
            raise ImportError("TransNetV2 package not installed")
        except Exception as e:
            logger.error(f"Error loading TransNetV2 model: {str(e)}")
            raise

        return (TransNet_model,)

    def _download_model(self, model_name, model_path):
        """
        Download TransNetV2 model from Hugging Face if not found locally.
        """
        try:
            os.makedirs(model_path, exist_ok=True)
            
            # Try to import huggingface_hub for downloading
            try:
                from huggingface_hub import hf_hub_download
                
                print(f"Downloading TransNetV2 model from Hugging Face...")
                
                # Download the PyTorch weights file
                weights_file = hf_hub_download(
                    repo_id="MiaoshouAI/transnetv2-pytorch-weights",
                    filename="transnetv2-pytorch-weights.pth",
                    local_dir=model_path,
                    local_dir_use_symlinks=False
                )
                
                logger.info(f"Successfully downloaded TransNetV2 weights to {weights_file}")
                
                # Create a marker file to indicate successful download
                marker_file = os.path.join(model_path, "model_ready.txt")
                with open(marker_file, "w") as f:
                    f.write(f"TransNetV2 model {model_name} downloaded successfully\n")
                    f.write(f"Downloaded from: MiaoshouAI/transnetv2-pytorch-weights\n")
                    f.write(f"Weights file: transnetv2-pytorch-weights.pth\n")
                
            except ImportError:
                logger.error("huggingface_hub not found. Please install: pip install huggingface_hub")
                logger.info("Alternatively, manually download transnetv2-pytorch-weights.pth from:")
                logger.info("https://huggingface.co/MiaoshouAI/transnetv2-pytorch-weights")
                logger.info(f"And place it in: {model_path}")
                
                # Create a marker file with manual download instructions
                marker_file = os.path.join(model_path, "model_ready.txt")
                with open(marker_file, "w") as f:
                    f.write(f"TransNetV2 model directory {model_name} ready\n")
                    f.write("Manual download required - huggingface_hub not available\n")
                    f.write("Download transnetv2-pytorch-weights.pth from:\n")
                    f.write("https://huggingface.co/MiaoshouAI/transnetv2-pytorch-weights\n")
                    f.write(f"And place it in: {model_path}\n")
                
                raise ImportError("huggingface_hub required for automatic download")
            
            except Exception as e:
                logger.error(f"Error downloading from Hugging Face: {str(e)}")
                logger.info("You can manually download the model from:")
                logger.info("https://huggingface.co/MiaoshouAI/transnetv2-pytorch-weights")
                logger.info(f"And place transnetv2-pytorch-weights.pth in: {model_path}")
                raise
            
        except Exception as e:
            logger.error(f"Error preparing model directory: {str(e)}")
            raise


class TransNetV2_Run:
    """
    A ComfyUI node for video scene segmentation using TransNetV2.
    Following the Qwen2.5-VL structure with optional video input.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "video": ("VIDEO",),
            },
            "required": {
                "TransNet_model": ("TRANSNET_MODEL",),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "min_scene_length": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 300,
                    "step": 1,
                    "display": "number"
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Leave empty for default temp directory"
                }),
            },
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("segment_paths", "path_string")
    FUNCTION = "TransNetV2_Run"
    CATEGORY = "MiaoshouAI Video Segmentation"

    def TransNetV2_Run(
        self,
        TransNet_model,
        threshold,
        min_scene_length,
        output_dir,
        video=None,
    ):
        if video is None:
            logger.error("No video input provided")
            return ([], "")
        
        # Handle video input - convert to temporary file if needed
        video_path = self._handle_video_input(video)
        if not video_path:
            logger.error("Failed to process video input")
            return ([], "")
        
        # Set up output directory
        if not output_dir:
            # Generate unique directory name using UUID
            import uuid
            unique_id = uuid.uuid4().hex[:8]  # Use first 8 chars for shorter name
            output_dir = os.path.join(folder_paths.temp_directory, f"transnet_segments_{unique_id}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Run TransNetV2 segmentation directly
            segment_paths = self._run_transnetv2_direct(
                TransNet_model,
                video_path,
                output_dir,
                threshold,
                min_scene_length
            )
            
            # Convert segment paths list to string format
            path_string = "\n".join(segment_paths) if segment_paths else ""
            
            logger.info(f"Successfully created {len(segment_paths)} video segments")
            return (segment_paths, path_string)
            
        except Exception as e:
            logger.error(f"Error in TransNetV2 segmentation: {str(e)}")
            return ([], "")
    
    def _handle_video_input(self, video):
        """
        Handle VIDEO input type and convert to file path.
        Based on Qwen2.5-VL temp_video function.
        """
        try:
            if VideoInput and isinstance(video, VideoInput):
                unique_id = uuid.uuid4().hex
                video_path = (
                    Path(folder_paths.temp_directory) / f"temp_video_{unique_id}.mp4"
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video.save_to(
                    str(video_path),
                    format="mp4",
                    codec="h264",
                )
                
                logger.info(f"Video saved to temporary path: {video_path}")
                return str(video_path)
            
            elif isinstance(video, str):
                if os.path.isfile(video):
                    return video
                else:
                    logger.error(f"Video file not found: {video}")
                    return None
            
            else:
                logger.warning(f"Unsupported video input type: {type(video)}")
                return None
                    
        except Exception as e:
            logger.error(f"Error handling video input: {str(e)}")
            return None

    def _run_transnetv2_direct(self, TransNet_model, video_path, output_dir, threshold, min_scene_length):
        """
        Run TransNetV2 segmentation directly using the loaded model.
        """
        try:
            import transnetv2_pytorch as transnetv2
            import cv2
            
            model = TransNet_model["model"]
            
            # Load video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video properties: {total_frames} frames, {fps} fps, {width}x{height}")
            
            # Read all frames
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            cap.release()
            
            if not frames:
                raise RuntimeError("No frames could be read from video")
            
            # Convert to numpy array
            frames_array = np.array(frames)
            
            # Run TransNetV2 prediction
            logger.info("Running TransNetV2 scene detection...")
            
            # Convert frames to tensor format - TransNetV2 expects uint8 [B, T, H, W, 3]
            # According to source: assert inputs.dtype == torch.uint8 and shape[2:] == [27, 48, 3]
            import torch.nn.functional as F
            from PIL import Image
            
            resized_frames = []
            for frame in frames_array:
                # Use PIL for resizing to maintain uint8 format
                pil_image = Image.fromarray(frame.astype(np.uint8))
                # Resize to 48x27 (width x height for PIL)
                resized_pil = pil_image.resize((48, 27), Image.BILINEAR)
                # Convert back to numpy array (uint8, HWC format)
                resized_frame = np.array(resized_pil, dtype=np.uint8)
                resized_frames.append(resized_frame)
            
            # Stack frames: (T, H, W, C) then add batch dim: (1, T, H, W, C)
            frames_array_resized = np.stack(resized_frames, axis=0)
            frames_array_batch = frames_array_resized[np.newaxis, ...]  # Add batch dimension
            
            # Convert to uint8 tensor (TransNetV2 requirement)
            frames_tensor = torch.from_numpy(frames_array_batch).to(dtype=torch.uint8)
            
            # Move to device
            frames_tensor = frames_tensor.to(TransNet_model["device"])
            
            logger.info(f"Input tensor shape: {frames_tensor.shape} (dtype: {frames_tensor.dtype})")
            
            # Run inference
            with torch.no_grad():
                predictions = model(frames_tensor)
                
                # Handle TransNetV2 output format
                if isinstance(predictions, tuple):
                    # Model returns (one_hot, {"many_hot": many_hot_predictions})
                    one_hot_predictions, many_hot_dict = predictions
                    logger.info(f"One-hot predictions shape: {one_hot_predictions.shape}")
                    logger.info(f"Many-hot predictions available: {list(many_hot_dict.keys())}")
                    single_frame_predictions = one_hot_predictions
                else:
                    # Model returns only one_hot predictions
                    logger.info(f"Predictions shape: {predictions.shape}")
                    single_frame_predictions = predictions
                
            # Convert predictions to numpy
            single_frame_predictions = single_frame_predictions.cpu().numpy().squeeze()
            all_frame_predictions = single_frame_predictions  # For compatibility
            
            # Find scene boundaries
            scenes = self._find_scenes(single_frame_predictions, threshold, min_scene_length)
            
            # Create video segments
            segment_paths = self._create_video_segments(
                video_path, scenes, output_dir, fps
            )
            # 删除第一个元素（如果列表非空）XUANCHEN 修改
            if segment_paths:  # 避免空列表切片报错
                segment_paths = segment_paths[1:]
                
            return segment_paths
            
        except Exception as e:
            logger.error(f"Error running TransNetV2 directly: {str(e)}")
            raise

    def _find_scenes(self, predictions, threshold, min_scene_length):
        """
        Find scene boundaries from TransNetV2 predictions.
        Uses the same algorithm as the reference project's predictions_to_scenes function.
        """
        # Convert predictions to binary (same as reference project)
        predictions_binary = (predictions > threshold).astype(np.uint8)
        
        scenes = []
        t, t_prev, start = -1, 0, 0
        
        for i, t in enumerate(predictions_binary):
            if t_prev == 1 and t == 0:  # Transition from scene boundary to normal frame - start of new scene
                start = i
            if t_prev == 0 and t == 1 and i != 0:  # Transition from normal frame to scene boundary - end of scene
                scenes.append([start, i])
            t_prev = t
            
        # Handle the last scene
        if t == 0:  # If last prediction is not a scene boundary
            scenes.append([start, len(predictions_binary)])
        
        # Handle case where all predictions are scene boundaries
        if len(scenes) == 0:
            return [(0, len(predictions_binary))]
        
        # Apply minimum scene length filtering
        filtered_scenes = []
        for start, end in scenes:
            if end - start >= min_scene_length:
                filtered_scenes.append((start, end))
            else:
                # If scene is too short, merge with previous scene or extend
                if filtered_scenes:
                    # Extend the previous scene to include this short one
                    prev_start, prev_end = filtered_scenes[-1]
                    filtered_scenes[-1] = (prev_start, end)
                else:
                    # If it's the first scene and too short, keep it anyway
                    filtered_scenes.append((start, end))
        
        return filtered_scenes if filtered_scenes else [(0, len(predictions_binary))]

    def _create_video_segments(self, video_path, scenes, output_dir, fps):
        """
        Create video segments based on scene boundaries using ffmpeg subprocess to preserve audio.
        """
        import subprocess
        
        segment_paths = []
        
        try:
            for i, (start_frame, end_frame) in enumerate(scenes):
                # Create output filename
                segment_filename = f"segment_{i+1:03d}.mp4"
                segment_path = os.path.join(output_dir, segment_filename)
                
                # Calculate time-based start and end times
                # Note: end_time is exclusive in both moviepy and ffmpeg -to parameter
                start_time = start_frame / fps
                end_time = end_frame / fps  # ffmpeg -to parameter is exclusive, matching moviepy's subclip behavior
                
                logger.info(f"Creating segment {i+1}: frames {start_frame}-{end_frame-1} (inclusive), time {start_time:.2f}s-{end_time:.2f}s")
                
                # Use ffmpeg subprocess to extract segment with audio preservation
                try:
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output files
                        '-ss', str(start_time),  # Start time
                        '-to', str(end_time),    # End time (exclusive)
                        '-i', video_path,        # Input file
                        '-c:v', 'libx264',       # Video codec
                        '-c:a', 'aac',           # Audio codec
                        '-preset', 'fast',       # Encoding speed
                        '-crf', '23',            # Quality (lower = better)
                        segment_path             # Output file
                    ]
                    
                    # Run ffmpeg command
                    result = subprocess.run(
                        ffmpeg_cmd,
                        capture_output=True,
                        text=True,
                        check=False  # Don't raise exception on non-zero exit
                    )
                    
                    # Check if command succeeded and file exists
                    if result.returncode == 0 and os.path.exists(segment_path):
                        # Get absolute path
                        abs_segment_path = os.path.abspath(segment_path)
                        segment_paths.append(abs_segment_path)
                        
                        logger.info(f"✅ Created segment {i+1}: {abs_segment_path} (duration: {end_time - start_time:.2f}s)")
                    else:
                        logger.error(f"❌ Failed to create segment {i+1}:")
                        logger.error(f"   Return code: {result.returncode}")
                        logger.error(f"   Error: {result.stderr}")
                        continue
                    
                except Exception as e:
                    logger.error(f"❌ Failed to create segment {i+1}: {str(e)}")
                    # Continue with next segment even if one fails
                    continue
        
        except Exception as e:
            logger.error(f"Error in video segmentation: {str(e)}")
            raise
        
        return segment_paths


class SelectVideo:
    """
    A ComfyUI node for selecting a specific video segment from TransNetV2 output.
    Takes a list of segment paths and returns the path at the specified index.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "segment_paths": ("LIST",),
            },
            "required": {
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selected_path",)
    FUNCTION = "select_video"
    CATEGORY = "MiaoshouAI Video Segmentation"

    def select_video(self, index, segment_paths=None):
        """
        Select a video segment path by index from the segment paths list.
        
        Args:
            index: The index of the segment to select (0-based)
            segment_paths: List of segment paths from TransNetV2_Run
            
        Returns:
            The path string at the specified index
        """
        if segment_paths is None or len(segment_paths) == 0:
            logger.warning("No segment paths provided")
            return ("",)
        
        # Validate index
        if index < 0:
            logger.warning(f"Index {index} is negative, using 0")
            index = 0
        elif index >= len(segment_paths):
            logger.warning(f"Index {index} is out of range (max: {len(segment_paths)-1}), using last index")
            index = len(segment_paths) - 1
        
        selected_path = segment_paths[index]
        logger.info(f"Selected segment {index}: {selected_path}")
        
        return (selected_path,)

#XuanChen 修改
class DeleteVideoSegment:
    """
    A ComfyUI node for deleting a specific video segment from a list of segments.
    Takes a list of segment paths and removes the element at the specified index,
    returning the updated list.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_paths": ("LIST",),
                "index_to_delete": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("updated_segment_paths",)
    FUNCTION = "delete_segment"
    CATEGORY = "MiaoshouAI Video Segmentation"

    def delete_segment(self, segment_paths, index_to_delete):
        """
        Delete a video segment path by index from the segment paths list.
        
        Args:
            segment_paths: List of segment paths to process
            index_to_delete: The index of the segment to delete (0-based)
            
        Returns:
            Updated list of segment paths after deletion
        """
        # Validate input list
        if not segment_paths:
            logger.warning("Empty segment paths list provided, returning empty list")
            return ([],)
        
        # Validate index range
        if index_to_delete < 0:
            logger.warning(f"Delete index {index_to_delete} is negative, no segment deleted")
            return (segment_paths.copy(),)
        if index_to_delete >= len(segment_paths):
            logger.warning(f"Delete index {index_to_delete} is out of range (max: {len(segment_paths)-1}), no segment deleted")
            return (segment_paths.copy(),)
        
        # Create a new list excluding the target index (avoid modifying original list)
        updated_paths = [path for i, path in enumerate(segment_paths) if i != index_to_delete]
        logger.info(f"Deleted segment at index {index_to_delete}, new list length: {len(updated_paths)}")
        
        return (updated_paths,)


class ZipCompress:
    """
    A ComfyUI node for compressing video segments into a zip file.
    Takes a list of segment paths and creates a compressed zip archive.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "images_or_video_path": ("STRING",),
            },
            "required": {
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "multiline": False,
                }),
                "image_format": (
                    ["PNG", "JPG", "WEBP", "MP4", "AVI", "MOV"],
                    {"default": "PNG"},
                ),
                "password": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional password for zip file"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("zip_filename",)
    FUNCTION = "compress_files"
    CATEGORY = "MiaoshouAI Video Segmentation"

    def compress_files(self, filename_prefix, image_format, password, images_or_video_path=None):
        """
        Compress video segments into a zip file.
        
        Args:
            filename_prefix: Prefix for the zip filename
            image_format: Format parameter (kept for interface compatibility)
            password: Optional password for the zip file
            images_or_video_path: String of file paths (newline-separated) to compress
            
        Returns:
            The path to the created zip file
        """
        import zipfile
        import datetime
        
        # Handle input - convert string to list if needed
        if images_or_video_path is None or images_or_video_path.strip() == "":
            logger.warning("No file paths provided for compression")
            return ("",)
        
        # Convert path string to list (TransNetV2_Run outputs newline-separated paths)
        if isinstance(images_or_video_path, str):
            file_paths = [path.strip() for path in images_or_video_path.split('\n') if path.strip()]
        else:
            file_paths = images_or_video_path
        
        if not file_paths:
            logger.warning("No valid file paths found")
            return ("",)
        
        # Create output directory if not exists
        # Use the directory of the first file as base for output
        first_file_dir = os.path.dirname(file_paths[0])
        
        # Generate zip filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{filename_prefix}_{timestamp}.zip"
        zip_path = os.path.join(first_file_dir, zip_filename)
        
        try:
            # Create zip file
            compression = zipfile.ZIP_DEFLATED
            compresslevel = 6  # Good balance between speed and compression
            
            with zipfile.ZipFile(zip_path, 'w', compression=compression, compresslevel=compresslevel) as zipf:
                # Set password if provided
                if password:
                    zipf.setpassword(password.encode('utf-8'))
                
                # Add each file to the zip
                for i, file_path in enumerate(file_paths):
                    if os.path.exists(file_path):
                        # Get just the filename for the archive
                        filename = os.path.basename(file_path)
                        
                        # Add file to zip
                        if password:
                            # For password-protected files, we need to use a different approach
                            zipf.write(file_path, filename)
                        else:
                            zipf.write(file_path, filename)
                        
                        logger.info(f"Added file {i+1}: {filename}")
                    else:
                        logger.warning(f"File not found, skipping: {file_path}")
            
            # Verify zip file was created
            if os.path.exists(zip_path):
                file_size = os.path.getsize(zip_path)
                logger.info(f"✅ Successfully created zip file: {zip_path}")
                logger.info(f"   File size: {file_size / (1024*1024):.2f} MB")
                logger.info(f"   Contains {len(file_paths)} files")
                if password:
                    logger.info("   Password protected: Yes")
                
                return (os.path.abspath(zip_path),)
            else:
                logger.error("Failed to create zip file")
                return ("",)
                
        except Exception as e:
            logger.error(f"Error creating zip file: {str(e)}")
            return ("",)


# Helper function similar to Qwen2.5-VL
def temp_video(video):
    """
    Create temporary video file from VideoInput.
    """
    unique_id = uuid.uuid4().hex
    video_path = (
        Path(folder_paths.temp_directory) / f"temp_video_{unique_id}.mp4"
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video.save_to(
        str(video_path),
        format="mp4",
        codec="h264",
    )

    return str(video_path)

class VideoConcatenator:
    """
    视频片段拼接节点：将多个视频片段按顺序拼接为单个视频（基于ffmpeg，无循环）
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_paths": ("LIST",),  # 从DeleteVideoSegment节点获取的剩余片段路径列表
                "output_filename": ("STRING", {
                    "default": "concatenated_video.mp4",
                    "placeholder": "拼接后的视频文件名（含扩展名）"
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "placeholder": "为空则使用第一个片段的目录"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "concatenate_videos"
    CATEGORY = "MiaoshouAI Video Segmentation"

    def concatenate_videos(self, segment_paths, output_filename, output_dir):
        import tempfile
        import subprocess

        # 校验输入片段
        if not segment_paths or len(segment_paths) < 1:
            logger.error("无有效视频片段可拼接")
            return ("",)

        # 确定输出目录
        if not output_dir:
            output_dir = os.path.dirname(segment_paths[0])
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        try:
            # 生成临时文件列表（ffmpeg拼接需要的格式）
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                for path in segment_paths:
                    # 路径需用引号包裹（避免含空格的路径出错）
                    f.write(f"file '{os.path.abspath(path)}'\n")
                temp_list_path = f.name

            # 用ffmpeg单命令拼接所有视频（无循环，底层高效处理）
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # 覆盖现有文件
                '-f', 'concat',  # 使用concat协议
                '-safe', '0',  # 允许绝对路径
                '-i', temp_list_path,  # 输入片段列表
                '-c', 'copy',  # 直接复制流（不重新编码，速度最快）
                output_path
            ]

            # 执行命令
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True
            )

            # 检查结果
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"视频拼接完成：{output_path}（{len(segment_paths)}个片段）")
                return (output_path,)
            else:
                logger.error(f"拼接失败：{result.stderr}")
                return ("",)

        finally:
            # 清理临时文件
            if 'temp_list_path' in locals() and os.path.exists(temp_list_path):
                os.remove(temp_list_path)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadTransNetModel": DownloadAndLoadTransNetModel,
    "TransNetV2_Run": TransNetV2_Run,
    "SelectVideo": SelectVideo,
    "ZipCompress": ZipCompress,
     "VideoConcatenator": VideoConcatenator  # 新增节点
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadTransNetModel": "🐾MiaoshouAI Load TransNet Model",
    "TransNetV2_Run": "🐾MiaoshouAI Segment Video",
    "SelectVideo": "🐾MiaoshouAI Select Video",
    "ZipCompress": "🐾MiaoshouAI Zip Compress",
    "VideoConcatenator": "🐾MiaoshouAI Concatenate Videos"  # 新增节点显示名
}

import os
import uuid
import folder_paths
import numpy as np
import logging
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
    Following the Qwen2.5-VL structure for consistency.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    [
                        "transnetv2-weights",
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
    CATEGORY = "TransNet"

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
        Create model directory structure. PyTorch weights should be pre-converted.
        """
        try:
            os.makedirs(model_path, exist_ok=True)
            
            # Create a marker file to indicate model directory is ready
            marker_file = os.path.join(model_path, "model_ready.txt")
            with open(marker_file, "w") as f:
                f.write(f"TransNetV2 model directory {model_name} ready\n")
                f.write("Note: PyTorch weights should be pre-converted and placed here\n")
                f.write("Expected file: transnetv2-pytorch-weights.pth\n")
            
            logger.info(f"TransNetV2 model directory created at {model_path}")
            
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
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("segment_paths",)
    FUNCTION = "TransNetV2_Run"
    CATEGORY = "TransNet"

    def TransNetV2_Run(
        self,
        TransNet_model,
        threshold,
        min_scene_length,
        output_dir,
        seed,
        video=None,
    ):
        if video is None:
            logger.error("No video input provided")
            return ([],)
        
        # Handle video input - convert to temporary file if needed
        video_path = self._handle_video_input(video, seed)
        if not video_path:
            logger.error("Failed to process video input")
            return ([],)
        
        # Set up output directory
        if not output_dir:
            output_dir = os.path.join(folder_paths.temp_directory, f"transnet_segments_{seed}")
        
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
            
            logger.info(f"Successfully created {len(segment_paths)} video segments")
            return (segment_paths,)
            
        except Exception as e:
            logger.error(f"Error in TransNetV2 segmentation: {str(e)}")
            return ([],)
    
    def _handle_video_input(self, video, seed):
        """
        Handle VIDEO input type and convert to file path.
        Based on Qwen2.5-VL temp_video function.
        """
        try:
            if VideoInput and isinstance(video, VideoInput):
                unique_id = uuid.uuid4().hex
                video_path = (
                    Path(folder_paths.temp_directory) / f"temp_video_{seed}_{unique_id}.mp4"
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
            
            return segment_paths
            
        except Exception as e:
            logger.error(f"Error running TransNetV2 directly: {str(e)}")
            raise

    def _find_scenes(self, predictions, threshold, min_scene_length):
        """
        Find scene boundaries from TransNetV2 predictions.
        """
        # Find frames where transition probability exceeds threshold
        transitions = np.where(predictions > threshold)[0]
        
        if len(transitions) == 0:
            return [(0, len(predictions))]
        
        # Group transitions and ensure minimum scene length
        scenes = []
        start_frame = 0
        
        for transition in transitions:
            if transition - start_frame >= min_scene_length:
                scenes.append((start_frame, transition))
                start_frame = transition
        
        # Add final scene
        if start_frame < len(predictions):
            scenes.append((start_frame, len(predictions)))
        
        return scenes

    def _create_video_segments(self, video_path, scenes, output_dir, fps):
        """
        Create video segments based on scene boundaries.
        """
        import cv2
        
        segment_paths = []
        
        # Open original video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        try:
            for i, (start_frame, end_frame) in enumerate(scenes):
                # Create output filename
                segment_filename = f"segment_{i+1:03d}.mp4"
                segment_path = os.path.join(output_dir, segment_filename)
                
                # Create video writer for this segment
                out = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))
                
                # Seek to start frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                # Write frames for this segment
                for frame_idx in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                
                out.release()
                
                # Get absolute path
                abs_segment_path = os.path.abspath(segment_path)
                segment_paths.append(abs_segment_path)
                
                logger.info(f"Created segment {i+1}: {abs_segment_path} (frames {start_frame}-{end_frame})")
        
        finally:
            cap.release()
        
        return segment_paths


# Helper function similar to Qwen2.5-VL
def temp_video(video: VideoInput, seed):
    """
    Create temporary video file from VideoInput.
    """
    unique_id = uuid.uuid4().hex
    video_path = (
        Path(folder_paths.temp_directory) / f"temp_video_{seed}_{unique_id}.mp4"
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video.save_to(
        str(video_path),
        format="mp4",
        codec="h264",
    )

    return str(video_path)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadTransNetModel": DownloadAndLoadTransNetModel,
    "TransNetV2_Run": TransNetV2_Run
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadTransNetModel": "Download and Load TransNet Model",
    "TransNetV2_Run": "TransNetV2 Run"
}
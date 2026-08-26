'''
Buff.py
Created by: https://github.com/BuffMcBigHuge
'''

import os
import io
import re
import random
import glob as glob_module
import folder_paths
import json
import time
import os.path
import fnmatch
from collections.abc import Mapping
import torch
import torch.nn.functional as torch_nn_func
import numpy as np
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
import torchvision.transforms.functional as F
from torchvision.utils import flow_to_image

class FilePathSelectorFromDirectory:
    """Selects a file from a directory based on specified file types, either randomly or sequentially."""
    
    # Class variable to store indices for sequential selection
    file_indices = {}
    
    # Class variable to cache file lists with timestamp
    file_cache = {}
    CACHE_EXPIRY = 60  # Cache expiry in seconds
    
    # Class variable to cache file counts for random selection
    file_count_cache = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {
                    "default": folder_paths.get_input_directory(),
                    "tooltip": "Path to the directory containing files to select from. Can be absolute or relative path."
                }),
                "file_types": ("STRING", {
                    "default": "mp4,mkv,webm",
                    "tooltip": "Comma-separated list of file extensions to include (e.g., 'mp4,jpg,png'). No dots needed."
                }),
                "selection_mode": (["randomize", "sequential"], {
                    "default": "randomize",
                    "tooltip": "randomize: Pick a random file each time (can be seeded)\nsequential: Go through files in order, remembering position between sessions"
                }),
                "include_subdirectories": (["True", "False"], {
                    "default": "True",
                    "tooltip": "When True, searches in subdirectories for matching files. When False, only searches in the specified directory."
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for random selection. Use the same seed to get the same file each time."
                }),
                "cache_duration": ("INT", {
                    "default": 60, 
                    "min": 0, 
                    "max": 3600,
                    "tooltip": "How long to cache the file list in seconds. 0 disables caching."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "select_file"
    CATEGORY = "utils"

    def get_file_count(self, directory_path, extensions, use_cache=True, cache_duration=None, include_subdirectories="False"):
        """Get the count of matching files in a directory without loading all filenames."""
        # Create a unique key for this directory + file types combination
        dir_key = f"{directory_path}_{','.join(extensions)}_{include_subdirectories}"
        
        # Use cache if available and not expired
        current_time = time.time()
        if use_cache and dir_key in self.file_count_cache:
            cache_entry = self.file_count_cache[dir_key]
            cache_time = cache_entry["timestamp"]
            expiry = cache_duration if cache_duration is not None else self.CACHE_EXPIRY
            
            if current_time - cache_time < expiry:
                return cache_entry["count"]
        
        # Count files efficiently
        count = 0
        if include_subdirectories == "True":
            for root, _, files in os.walk(directory_path):
                for ext in extensions:
                    pattern = f"*{ext}"
                    count += len(fnmatch.filter(files, pattern))
        else:
            for ext in extensions:
                pattern = f"*{ext}"
                count += len(fnmatch.filter([entry.name for entry in os.scandir(directory_path) if entry.is_file()], pattern))
        
        # Update cache
        self.file_count_cache[dir_key] = {
            "count": count,
            "timestamp": current_time
        }
        
        return count

    def get_file_at_index(self, directory_path, extensions, index, include_subdirectories="False"):
        """Get the file at the specified index from the directory with matching extensions."""
        files = self.get_matching_files(directory_path, extensions, include_subdirectories=include_subdirectories)
        
        if not files or index >= len(files):
            return None
        
        return files[index]

    def select_random_file(self, directory_path, extensions, seed=None, include_subdirectories="False"):
        """Select a random file from the directory with matching extensions."""
        # Get the count of matching files
        count = self.get_file_count(directory_path, extensions, include_subdirectories=include_subdirectories)
        
        if count == 0:
            print(f"No files with extensions {extensions} found in {directory_path}")
            return None
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Select a random index
        index = random.randint(0, count - 1)
        
        # Get the file at that index
        return self.get_file_at_index(directory_path, extensions, index, include_subdirectories=include_subdirectories)

    def get_matching_files(self, directory_path, extensions, use_cache=True, cache_duration=None, include_subdirectories="False"):
        """Get all files with matching extensions in the directory."""
        # Ensure directory_path is properly formatted
        directory_path = directory_path.strip('"\'')
        
        if not os.path.isdir(directory_path):
            print(f"Warning: {directory_path} is not a valid directory")
            return []
        
        # Convert string "True"/"False" to boolean
        include_subdirs = include_subdirectories.lower() == "true"
        
        # Check if we have a valid cached result
        cache_key = f"{directory_path}_{','.join(extensions)}_{include_subdirs}"
        current_time = time.time()
        
        if use_cache and cache_key in self.file_cache:
            cache_time, files = self.file_cache[cache_key]
            if cache_duration is None:
                cache_duration = self.CACHE_EXPIRY
            if current_time - cache_time < cache_duration:
                return files
        
        matching_files = []
        
        try:
            # Walk through directory and subdirectories if include_subdirs is True
            if include_subdirs:
                for root, _, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if any(file.lower().endswith(f".{ext.lower().strip()}") for ext in extensions):
                            matching_files.append(file_path)
            else:
                # Original behavior - only search in the specified directory
                for file in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, file)
                    if os.path.isfile(file_path) and any(file.lower().endswith(f".{ext.lower().strip()}") for ext in extensions):
                        matching_files.append(file_path)
            
            # Debug output
            if not matching_files:
                print(f"No files with extensions {extensions} found in {directory_path} (include_subdirs={include_subdirs})")
            else:
                print(f"Found {len(matching_files)} files with extensions {extensions} in {directory_path}")
        
        except Exception as e:
            print(f"Error searching for files in {directory_path}: {str(e)}")
            return []
        
        # Update cache
        self.file_cache[cache_key] = (current_time, matching_files)
        
        return matching_files

    def select_sequential_file(self, directory_path, extensions, dir_key, include_subdirectories="False"):
        """Select the next file in sequence from the directory with matching extensions."""
        # Get all matching files
        files = self.get_matching_files(directory_path, extensions, include_subdirectories=include_subdirectories)
        
        if not files:
            print(f"No files with extensions {extensions} found in {directory_path}")
            return None
        
        # Get the current index for this directory, or initialize to 0
        if dir_key not in self.file_indices:
            self.file_indices[dir_key] = 0
        
        # Get the file at the current index
        file_path = files[self.file_indices[dir_key]]
        
        # Increment the index for next time, wrapping around if necessary
        self.file_indices[dir_key] = (self.file_indices[dir_key] + 1) % len(files)
        
        # Save the updated indices
        self.save_indices()
        
        return file_path

    def select_file(self, directory_path, file_types, selection_mode, include_subdirectories="False", seed=None, cache_duration=60):
        """Select a file from the directory based on the specified selection mode."""
        # Strip quotation marks from directory path if present
        directory_path = directory_path.strip('"\'')
        
        # Ensure directory_path is a valid directory
        if not os.path.isdir(directory_path):
            print(f"Error: {directory_path} is not a valid directory")
            return ("",)
        
        # Clear cache if needed to ensure fresh results
        self.clear_cache()
        
        extensions = [ext.strip() for ext in file_types.split(',')]
        
        if selection_mode == "randomize":
            result = self.select_random_file(directory_path, extensions, seed, include_subdirectories=include_subdirectories)
        else:  # sequential
            dir_key = f"{directory_path}_{file_types}_{include_subdirectories}"
            result = self.select_sequential_file(directory_path, extensions, dir_key, include_subdirectories=include_subdirectories)
        
        # Debug output - show raw path without quotes
        print(f"Selected file: {result}")
        
        # Return raw path without any quoting
        return (result if result else "",) 

    # Optional: Clear cache method
    @classmethod
    def clear_cache(cls):
        cls.file_cache = {}
        cls.file_count_cache = {}

    # Optional: Save indices to persist between ComfyUI sessions
    @classmethod
    def save_indices(cls):
        try:
            with open("file_selector_indices.json", "w") as f:
                json.dump(cls.file_indices, f)
        except Exception as e:
            print(f"Error saving file indices: {e}")
    
    # Optional: Load indices when ComfyUI starts
    @classmethod
    def load_indices(cls):
        try:
            if os.path.exists("file_selector_indices.json"):
                with open("file_selector_indices.json", "r") as f:
                    cls.file_indices = json.load(f)
        except Exception as e:
            print(f"Error loading file indices: {e}")
            cls.file_indices = {}

# Try to load saved indices
FilePathSelectorFromDirectory.load_indices()

class StringProcessor:
    """Processes a string with various operations like slicing to extract first or last parts."""
    
    @classmethod
    def INPUT_TYPES(cls):
        operations = [
            "first_n_chars", 
            "last_n_chars", 
            "first_n_words", 
            "last_n_words",
            "custom_expression"
        ]
        
        operations_info = {
            "first_n_chars": "Extract the first N characters from the input string",
            "last_n_chars": "Extract the last N characters from the input string",
            "first_n_words": "Extract the first N words from the input string (using delimiter)",
            "last_n_words": "Extract the last N words from the input string (using delimiter)",
            "custom_expression": "Use a custom Python expression to manipulate the string"
        }
        
        return {
            "required": {
                "input_string": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "tooltip": "The input string to process. Can be multiple lines."
                }),
                "operation": (operations, {
                    "default": "first_n_chars",
                    "tooltip": "Select the operation to perform on the input string:\n" + 
                              "\n".join([f"• {op}: {operations_info[op]}" for op in operations])
                }),
                "n": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 10000,
                    "tooltip": "Number of characters or words to extract"
                }),
            },
            "optional": {
                "delimiter": ("STRING", {
                    "default": " ",
                    "tooltip": "Character(s) used to separate words (only used for word operations)"
                }),
                "custom_expression": ("STRING", {
                    "default": "{s}.upper()",
                    "multiline": True,
                    "tooltip": """Custom Python expression to manipulate the input string.
Use {s} as a placeholder for the input string.

Examples:
• {s}.upper() - Convert to uppercase
• {s}.lower() - Convert to lowercase
• {s}.replace('old', 'new') - Replace text
• {s}[::-1] - Reverse the string
• {s}.split(',')[0] - Get first item of CSV
• ' '.join([w.capitalize() for w in {s}.split()]) - Title case
• {s}.strip() - Remove whitespace from ends
• {s} + ' (modified)' - Append text"""
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_string",)
    FUNCTION = "process_string"
    CATEGORY = "utils"

    def process_string(self, input_string, operation, n, delimiter=" ", custom_expression="{s}.upper()"):
        if not input_string:
            return ("",)
            
        if operation == "first_n_chars":
            result = input_string[:n]
        elif operation == "last_n_chars":
            result = input_string[-n:]
        elif operation == "first_n_words":
            words = input_string.split(delimiter)
            result = delimiter.join(words[:n])
        elif operation == "last_n_words":
            words = input_string.split(delimiter)
            result = delimiter.join(words[-n:])
        elif operation == "custom_expression":
            try:
                # Replace {s} with the actual string (properly escaped)
                s = input_string  # Define s as a variable
                formatted_expr = custom_expression.replace("{s}", "s")
                
                # Evaluate the expression in a controlled environment
                # We only allow access to the input string and string methods
                result = eval(formatted_expr)
                
                # Handle non-string results by converting them
                if not isinstance(result, str):
                    result = str(result)
            except Exception as e:
                result = f"Error in custom expression: {str(e)}"
                print(f"StringProcessor error: {str(e)} in expression '{custom_expression}'")
        else:
            result = input_string
            
        return (result,)


class ConsoleOutput:
    """Displays text in the console and acts as an output node for ComfyUI workflows."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "tooltip": "Text to display in the console. Can be multiple lines."
                }),
                "prefix": ("STRING", {
                    "default": "ConsoleOutput",
                    "tooltip": "Prefix to add before the text in console output. Helps identify different console outputs."
                }),
            },
            "optional": {
                "show_timestamp": (["True", "False"], {
                    "default": "True",
                    "tooltip": "When True, includes a timestamp with each console output. When False, shows only the text."
                }),
            }
        }

    RETURN_TYPES = ()  # No outputs as this is an output node
    OUTPUT_NODE = True  # Mark as an output node
    FUNCTION = "output_to_console"
    CATEGORY = "utils"

    def output_to_console(self, text, prefix="ConsoleOutput", show_timestamp="True"):
        timestamp = ""
        if show_timestamp == "True":
            timestamp = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            
        formatted_prefix = f"[{prefix}] " if prefix else ""
        print(f"{timestamp}{formatted_prefix}{text}")
        
        return ()  # Return empty tuple as this is an output node


class TwoImageConcatenator:
    """Concatenates two 3-channel images into a single 6-channel image."""
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Define the input types for the node
        return {
            "required": {
                # image1: Connect the PREVIOUS PROCESSED FRAME here for temporal models
                "image1": ("IMAGE", {
                    "tooltip": "First input image (3 channels). Typically the previous frame for temporal models."
                }),
                # image2: Connect the CURRENT UNPROCESSED FRAME here for temporal models
                "image2": ("IMAGE", {
                    "tooltip": "Second input image (3 channels). Typically the current frame for temporal models."
                }),
            },
        }

    # Define the return types of the node
    RETURN_TYPES = ("IMAGE",)
    # Define the name of the function that will be executed when the node runs
    FUNCTION = "concatenate_images"
    # Define the category the node will appear under in the ComfyUI menu
    CATEGORY = "Image" # You can change this category if you like

    def concatenate_images(self, image1, image2):
        """
        Concatenates two image tensors along the channel dimension.

        Args:
            image1 (torch.Tensor): The first input image tensor (batch, height, width, channels=3).
            image2 (torch.Tensor): The second input image tensor (batch, height, width, channels=3).

        Returns:
            torch.Tensor: A single tensor with channels from both images (batch, height, width, channels=6).
        """
        # ComfyUI images are typically [batch, height, width, channels]
        # We need to concatenate along the last dimension (channels)

        # Ensure both images have the same dimensions except for channels
        if image1.shape[0] != image2.shape[0] or \
           image1.shape[1] != image2.shape[1] or \
           image1.shape[2] != image2.shape[2]:
            raise ValueError("Input images must have the same batch size, height, and width.")

        # Check if input images are 3 channels
        if image1.shape[3] != 3 or image2.shape[3] != 3:
             raise ValueError("Both input images must be 3-channel images.")


        # Concatenate the tensors along the channel dimension (dimension 3)
        # The result will be [batch, height, width, 6]
        concatenated_image = torch.cat((image1, image2), dim=3)

        # Return the resulting 6-channel image tensor
        return (concatenated_image,)


class RaftOpticalFlowNode:
    """
    A custom ComfyUI node to calculate optical flow between two images using a RAFT model.
    """
    def __init__(self):
        # Load the pre-trained RAFT model
        # Using Raft_Large_Weights.DEFAULT gets the best available weights
        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights, progress=True).eval()
        # Get the preprocessing transform required by the model
        # This transform is designed to take a list of PIL images or a tensor [N, 3, H, W]
        # and return tensors ready for the model.
        self.preprocess = self.weights.transforms()

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                "image_a": ("IMAGE", {
                    "tooltip": "First image (previous frame). Must be same size as image_b."
                }),
                "image_b": ("IMAGE", {
                    "tooltip": "Second image (current frame). Must be same size as image_a."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",) # The node outputs an image (optical flow visualization)
    FUNCTION = "calculate_flow" # The name of the function that will be executed
    CATEGORY = "Image" # The category under which the node will appear in ComfyUI

    def pad_to_multiple_of_8(self, tensor):
        """Pad tensor to dimensions divisible by 8."""
        batch_size, height, width, channels = tensor.shape
        
        # Calculate padding needed
        pad_h = (8 - (height % 8)) % 8
        pad_w = (8 - (width % 8)) % 8
        
        if pad_h == 0 and pad_w == 0:
            return tensor, (0, 0, 0, 0)  # No padding needed
            
        # Pad the tensor
        # Format: (left, right, top, bottom, front, back)
        # This ensures we only pad the spatial dimensions (H, W), not the channel dimension
        padding = (0, 0, 0, pad_w, 0, pad_h)
        padded_tensor = torch.nn.functional.pad(tensor, padding, mode='replicate')
        
        return padded_tensor, (0, pad_w, 0, pad_h)  # Return padding in (left, right, top, bottom) format

    def crop_to_original(self, tensor, padding):
        """Crop tensor back to original dimensions."""
        if padding == (0, 0, 0, 0):
            return tensor
            
        left, right, top, bottom = padding
        if right == 0 and bottom == 0:
            return tensor
            
        return tensor[:, :-bottom if bottom else None, :-right if right else None, :]

    def calculate_flow(self, image_a: torch.Tensor, image_b: torch.Tensor):
        """
        Calculates the optical flow between image_a and image_b.
        Handles images that are not divisible by 8 by padding and cropping.
        """
        # Get the device from the input tensors (ComfyUI will handle device placement)
        device = image_a.device
        batch_size = image_a.size(0)
        
        # Store original shapes for later cropping
        original_shapes = [(img.shape[1], img.shape[2]) for img in image_a]
        
        # Pad images to dimensions divisible by 8
        padded_a, padding_a = self.pad_to_multiple_of_8(image_a)
        padded_b, padding_b = self.pad_to_multiple_of_8(image_b)
        
        processed_img1_list = []
        processed_img2_list = []

        # Process each image pair in the batch individually
        for i in range(batch_size):
            # Convert ComfyUI tensor (H, W, C, 0-1) to PIL Image (ready for preprocess)
            # Note: ComfyUI format is [B, H, W, C], we need [B, C, H, W] for preprocessing
            img1_pil = F.to_pil_image(padded_a[i].permute(2, 0, 1).cpu())  # Move to CPU for PIL conversion
            img2_pil = F.to_pil_image(padded_b[i].permute(2, 0, 1).cpu())  # Move to CPU for PIL conversion

            try:
                # Pass PIL images directly to preprocess
                img1_processed_single, img2_processed_single = self.preprocess(img1_pil, img2_pil)
                
                # Ensure the processed tensors have shape [1, C, H, W] and move to correct device
                if len(img1_processed_single.shape) == 3:  # If shape is [C, H, W]
                    img1_processed_single = img1_processed_single.unsqueeze(0)  # Add batch dimension
                if len(img2_processed_single.shape) == 3:  # If shape is [C, H, W]
                    img2_processed_single = img2_processed_single.unsqueeze(0)  # Add batch dimension
                
                # Move tensors to the same device as input
                img1_processed_single = img1_processed_single.to(device)
                img2_processed_single = img2_processed_single.to(device)
                
                processed_img1_list.append(img1_processed_single)
                processed_img2_list.append(img2_processed_single)
            except Exception as e:
                print(f"Error during preprocessing batch item {i}: {e}")
                continue

        if not processed_img1_list or not processed_img2_list:
            print("Preprocessing failed for all items in the batch.")
            h, w = padded_a.size(1), padded_a.size(2)
            return (torch.zeros((batch_size, h, w, 3), dtype=torch.float32, device=device),)

        # Concatenate the processed single tensors back into batches
        # Each tensor should be [1, C, H, W], concatenating gives [B, C, H, W]
        img1_processed = torch.cat(processed_img1_list, dim=0)  # Shape [Batch, C, H, W]
        img2_processed = torch.cat(processed_img2_list, dim=0)  # Shape [Batch, C, H, W]

        # Print shapes for debugging
        print(f"Processed image shapes - img1: {img1_processed.shape}, img2: {img2_processed.shape}")

        # Move model to the same device as input tensors
        self.model = self.model.to(device)

        # Perform inference
        with torch.no_grad():
            list_of_flows = self.model(img1_processed, img2_processed)
            predicted_flow = list_of_flows[-1]  # Shape [Batch, 2, H, W]

        # Convert flow to visualization using torchvision's flow_to_image
        # flow_to_image expects input on CPU and returns uint8 [0,255]
        flow_vis = flow_to_image(predicted_flow.cpu())  # [B, 3, H, W], uint8

        # Convert to float32 and scale to [0,1]
        flow_vis = flow_vis.float() / 255.0

        # Move back to the original device and convert to ComfyUI format [B, H, W, C]
        flow_vis = flow_vis.to(device).permute(0, 2, 3, 1).contiguous()
        
        # Crop back to original dimensions
        flow_vis = self.crop_to_original(flow_vis, padding_a)

        return (flow_vis,)


class FrameRateModulator:
    """Modulates the frame rate of an image sequence by resampling frames to achieve target frame count or multiplier."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input batch of images to resample"
                }),
                "mode": (["frame_count", "multiplier"], {
                    "default": "frame_count",
                    "tooltip": "frame_count: Set exact number of output frames\nmultiplier: Scale input frame count by a factor"
                }),
            },
            "optional": {
                "target_frame_count": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Target number of frames for output (used when mode is 'frame_count')"
                }),
                "frame_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Multiplier for frame count (used when mode is 'multiplier'). 2.0 = double frames (slow motion), 0.5 = half frames (speed up)"
                }),
                "interpolation": (["nearest", "linear", "cubic"], {
                    "default": "linear",
                    "tooltip": "Interpolation method:\nnearest: No interpolation, duplicate/skip frames\nlinear: Linear blending between adjacent frames\ncubic: Smooth cubic interpolation"
                }),
                "loop_mode": (["clamp", "repeat", "mirror"], {
                    "default": "clamp",
                    "tooltip": "How to handle out-of-bounds frames:\nclamp: Use first/last frame\nrepeat: Loop the sequence\nmirror: Reverse and repeat"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    FUNCTION = "modulate_framerate"
    CATEGORY = "Image/Animation"

    def get_frame_at_position(self, images, position, loop_mode):
        """Get frame at a specific position, handling out-of-bounds with loop_mode."""
        total_frames = images.shape[0]
        
        if loop_mode == "clamp":
            # Clamp to valid range
            idx = max(0, min(total_frames - 1, int(position)))
            return images[idx]
        elif loop_mode == "repeat":
            # Loop the sequence
            idx = int(position) % total_frames
            return images[idx]
        elif loop_mode == "mirror":
            # Mirror the sequence
            cycle_length = (total_frames - 1) * 2
            pos_in_cycle = int(position) % cycle_length
            if pos_in_cycle < total_frames:
                return images[pos_in_cycle]
            else:
                mirror_idx = cycle_length - pos_in_cycle
                return images[mirror_idx]
        
        return images[0]  # fallback

    def interpolate_frames(self, images, positions, interpolation, loop_mode):
        """Interpolate frames at given positions."""
        device = images.device
        dtype = images.dtype
        batch_size, height, width, channels = images.shape
        output_frames = len(positions)
        
        result = torch.zeros((output_frames, height, width, channels), dtype=dtype, device=device)
        
        for i, pos in enumerate(positions):
            if interpolation == "nearest":
                # Simple nearest neighbor
                result[i] = self.get_frame_at_position(images, round(pos), loop_mode)
            
            elif interpolation == "linear":
                # Linear interpolation between adjacent frames
                pos_floor = int(torch.floor(torch.tensor(pos)).item())
                pos_ceil = pos_floor + 1
                alpha = pos - pos_floor
                
                frame_a = self.get_frame_at_position(images, pos_floor, loop_mode)
                frame_b = self.get_frame_at_position(images, pos_ceil, loop_mode)
                
                result[i] = frame_a * (1 - alpha) + frame_b * alpha
            
            elif interpolation == "cubic":
                # Cubic interpolation using 4 points
                pos_int = int(pos)
                alpha = pos - pos_int
                
                # Get 4 surrounding frames
                p0 = self.get_frame_at_position(images, pos_int - 1, loop_mode)
                p1 = self.get_frame_at_position(images, pos_int, loop_mode)
                p2 = self.get_frame_at_position(images, pos_int + 1, loop_mode)
                p3 = self.get_frame_at_position(images, pos_int + 2, loop_mode)
                
                # Cubic interpolation formula
                a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
                b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
                c = -0.5 * p0 + 0.5 * p2
                d = p1
                
                result[i] = a * (alpha ** 3) + b * (alpha ** 2) + c * alpha + d
                
                # Clamp values to valid range
                result[i] = torch.clamp(result[i], 0.0, 1.0)
        
        return result

    def modulate_framerate(self, images, mode, target_frame_count=30, frame_multiplier=1.0, 
                          interpolation="linear", loop_mode="clamp"):
        """
        Modulate the frame rate of an image sequence.
        
        Args:
            images: Input batch of images [batch, height, width, channels]
            mode: "frame_count" or "multiplier"
            target_frame_count: Target number of output frames
            frame_multiplier: Multiplier for frame count
            interpolation: Interpolation method ("nearest", "linear", "cubic")
            loop_mode: How to handle out-of-bounds ("clamp", "repeat", "mirror")
        
        Returns:
            tuple: (resampled_images, output_frame_count)
        """
        input_frame_count = images.shape[0]
        
        # Determine output frame count
        if mode == "frame_count":
            output_frame_count = target_frame_count
        else:  # multiplier
            output_frame_count = max(1, int(input_frame_count * frame_multiplier))
        
        print(f"FrameRateModulator: {input_frame_count} → {output_frame_count} frames "
              f"(mode: {mode}, interpolation: {interpolation})")
        
        # If no change needed, return original
        if output_frame_count == input_frame_count:
            return (images, input_frame_count)
        
        # Calculate positions for resampling
        if output_frame_count == 1:
            # Special case: single frame output
            positions = [input_frame_count // 2]
        else:
            # Map output frames to input positions
            positions = []
            for i in range(output_frame_count):
                # Map from [0, output_frame_count-1] to [0, input_frame_count-1]
                pos = i * (input_frame_count - 1) / (output_frame_count - 1)
                positions.append(pos)
        
        # Perform interpolation
        resampled_images = self.interpolate_frames(images, positions, interpolation, loop_mode)
        
        return (resampled_images, output_frame_count)


class MultilineTextSplitter:
    """Splits multiline text into multiple outputs based on specified parameters."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Multiline text input to split into multiple outputs"
                }),
                "num_outputs": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 20,
                    "tooltip": "Number of output text fields to create"
                }),
                "lines_per_output": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,
                    "tooltip": "Number of lines per output (used when split_evenly is False)"
                }),
                "split_evenly": (["True", "False"], {
                    "default": "False",
                    "tooltip": "When True, splits lines evenly across outputs. When False, uses lines_per_output for each output sequentially."
                }),
            },
            "optional": {
                "remove_empty_lines": (["True", "False"], {
                    "default": "False",
                    "tooltip": "When True, removes empty lines from the input before processing. When False, empty lines are preserved."
                }),
            }
        }
    
    # The num_outputs parameter controls how many outputs contain data
    # Unused outputs will return empty strings
    RETURN_TYPES = tuple(["STRING"] * 20)  # Maximum 20 outputs
    RETURN_NAMES = tuple([f"text_{i+1}" for i in range(20)])  # Return names for all outputs
    FUNCTION = "split_text"
    CATEGORY = "utils"
    
    def split_text(self, input_text, num_outputs, lines_per_output, split_evenly, remove_empty_lines="False"):
        """
        Splits multiline text into multiple outputs.
        
        Args:
            input_text: The multiline text input
            num_outputs: Number of outputs to create
            lines_per_output: Lines per output when split_evenly is False
            split_evenly: Whether to split evenly across outputs
            remove_empty_lines: Whether to remove empty lines before processing
        
        Returns:
            Tuple of strings, one for each output (up to 20 outputs)
        """
        # Split input into lines
        lines = input_text.split('\n')
        
        # Remove empty lines if requested
        if remove_empty_lines == "True":
            lines = [line for line in lines if line.strip() != ""]
        
        total_lines = len(lines)
        
        # Ensure num_outputs doesn't exceed maximum
        num_outputs = min(num_outputs, 20)
        
        # Initialize results list with empty strings for all possible outputs
        results = [""] * 20
        
        if total_lines == 0:
            # Return empty strings for all outputs
            return tuple(results)
        
        if split_evenly == "True":
            # Split evenly across outputs
            if num_outputs == 0:
                return tuple(results)
            
            # Calculate how many lines each output should get
            lines_per_output_actual = total_lines // num_outputs
            remainder = total_lines % num_outputs
            
            start_idx = 0
            for i in range(num_outputs):
                # Distribute remainder lines across first outputs
                current_lines = lines_per_output_actual + (1 if i < remainder else 0)
                end_idx = start_idx + current_lines
                output_text = '\n'.join(lines[start_idx:end_idx])
                results[i] = output_text
                start_idx = end_idx
        else:
            # Use lines_per_output for each output sequentially
            start_idx = 0
            for i in range(num_outputs):
                end_idx = start_idx + lines_per_output
                if start_idx >= total_lines:
                    # No more lines to process
                    results[i] = ""
                else:
                    # Get lines for this output (up to end_idx or end of lines)
                    output_lines = lines[start_idx:end_idx]
                    output_text = '\n'.join(output_lines)
                    results[i] = output_text
                start_idx = end_idx
        
        return tuple(results)


class TextTemplateCombiner:
    """Combines string inputs into one text output using numbered placeholders."""

    MAX_INPUTS = 20
    PLACEHOLDER_PATTERN = re.compile(r"(?<!\{)\{([1-9]\d*)\}(?!\})")
    CHOICE_PATTERN = re.compile(r"(?<!\{)\{([^{}\n]*\|[^{}\n]*)\}(?!\})")

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for index in range(1, cls.MAX_INPUTS + 1):
            optional[f"text_{index}"] = ("STRING", {
                "default": "",
                "forceInput": True,
                "tooltip": f"String inserted when the main text contains {{{index}}}."
            })

        return {
            "required": {
                "main": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Primary text template. Use {1}, {2}, etc. to insert connected text inputs. Use {foo|bar} to choose a random option."
                }),
                "inputcount": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": cls.MAX_INPUTS,
                    "step": 1,
                    "tooltip": "Number of optional text inputs to show. The browser UI also increases this from placeholders in main."
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "combine_text"
    CATEGORY = "utils"

    @staticmethod
    def _stringify(value):
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @classmethod
    def _get_input_reference_index(cls, value):
        value = value.strip()
        if not re.fullmatch(r"[1-9]\d*", value):
            return None
        index = int(value)
        if index > cls.MAX_INPUTS:
            return None
        return index

    @classmethod
    def IS_CHANGED(cls, main, inputcount=1, **kwargs):
        if cls.CHOICE_PATTERN.search(cls._stringify(main)):
            return float("nan")
        return ""

    def combine_text(self, main, inputcount=1, **kwargs):
        main = self._stringify(main)
        try:
            inputcount = int(inputcount)
        except (TypeError, ValueError):
            inputcount = 1
        inputcount = max(0, min(inputcount, self.MAX_INPUTS))

        placeholder_indexes = [
            int(match.group(1))
            for match in self.PLACEHOLDER_PATTERN.finditer(main)
            if int(match.group(1)) <= self.MAX_INPUTS
        ]
        for choice_match in self.CHOICE_PATTERN.finditer(main):
            for option in choice_match.group(1).split("|"):
                index = self._get_input_reference_index(option)
                if index is not None:
                    placeholder_indexes.append(index)

        active_count = max([inputcount] + placeholder_indexes) if placeholder_indexes else inputcount

        replacements = {
            str(index): self._stringify(kwargs.get(f"text_{index}", ""))
            for index in range(1, active_count + 1)
        }

        def resolve_option(option):
            option = option.strip()
            index = self._get_input_reference_index(option)
            if index is not None:
                return replacements.get(str(index), "")
            return option

        def replace_choice(match):
            options = match.group(1).split("|")
            if not options:
                return ""
            return resolve_option(random.choice(options))

        def replace_placeholder(match):
            index = match.group(1)
            if int(index) > self.MAX_INPUTS:
                return match.group(0)
            return replacements.get(index, "")

        text = self.CHOICE_PATTERN.sub(replace_choice, main)
        return (self.PLACEHOLDER_PATTERN.sub(replace_placeholder, text),)


class BatchRaftOpticalFlowNode:
    """
    Computes RAFT optical flow on consecutive pairs from an image batch.
    Given n images, produces n-1 flow visualizations: (0,1), (1,2), ..., (n-2,n-1).
    Uses CUDA when available and processes pairs in chunks to avoid OOM.
    """
    def __init__(self):
        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights, progress=True).eval()
        self.preprocess = self.weights.transforms()
        self._device = None

    def _get_device(self):
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Batch of images [B, H, W, C]. Optical flow is computed between each consecutive pair."
                }),
            },
            "optional": {
                "batch_chunk_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Number of pairs to process per GPU batch. Lower values use less VRAM. Increase for speed if you have enough VRAM."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("optical_flow",)
    FUNCTION = "calculate_batch_flow"
    CATEGORY = "Image"

    def _pad_to_multiple_of_8(self, height, width):
        """Return (pad_h, pad_w) needed to make dimensions divisible by 8."""
        pad_h = (8 - (height % 8)) % 8
        pad_w = (8 - (width % 8)) % 8
        return pad_h, pad_w

    def calculate_batch_flow(self, images: torch.Tensor, batch_chunk_size: int = 4):
        batch_size = images.shape[0]
        if batch_size < 2:
            raise ValueError("BatchRaftOpticalFlowNode requires at least 2 images in the batch.")

        device = self._get_device()
        self.model = self.model.to(device)

        num_pairs = batch_size - 1
        print(f"BatchRaftOpticalFlowNode: {num_pairs} pair(s), device={device}, chunk_size={batch_chunk_size}")

        _, orig_h, orig_w, _ = images.shape
        pad_h, pad_w = self._pad_to_multiple_of_8(orig_h, orig_w)

        # [B, H, W, C] -> [B, C, H, W] as float32, all at once on CPU
        imgs_chw = images.permute(0, 3, 1, 2).contiguous()

        if pad_h > 0 or pad_w > 0:
            imgs_chw = torch.nn.functional.pad(imgs_chw, (0, pad_w, 0, pad_h), mode='replicate')

        # Build consecutive pair tensors without PIL conversion:
        # frames_a = [0, 1, ..., n-2], frames_b = [1, 2, ..., n-1]
        frames_a = imgs_chw[:-1]  # [n-1, C, H, W]
        frames_b = imgs_chw[1:]   # [n-1, C, H, W]

        flow_vis_list = []

        for chunk_start in range(0, num_pairs, batch_chunk_size):
            chunk_end = min(chunk_start + batch_chunk_size, num_pairs)

            chunk_a = frames_a[chunk_start:chunk_end].to(device)
            chunk_b = frames_b[chunk_start:chunk_end].to(device)

            # RAFT preprocess: expects [B,C,H,W] tensors in 0-255 uint8 range or 0-1 float
            # The transforms() call normalizes and converts — works on tensor batches directly
            chunk_a_proc, chunk_b_proc = self.preprocess(chunk_a, chunk_b)

            with torch.no_grad():
                flows = self.model(chunk_a_proc, chunk_b_proc)
                predicted_flow = flows[-1]  # [chunk, 2, H, W]

            chunk_vis = flow_to_image(predicted_flow.cpu()).float() / 255.0
            flow_vis_list.append(chunk_vis)

            # Free GPU memory between chunks
            del chunk_a, chunk_b, chunk_a_proc, chunk_b_proc, flows, predicted_flow

        # Concatenate all chunks: [n-1, 3, H, W]
        all_flow_vis = torch.cat(flow_vis_list, dim=0)

        # Convert to ComfyUI format [B, H, W, C] and crop padding
        all_flow_vis = all_flow_vis.permute(0, 2, 3, 1).contiguous()
        if pad_h > 0 or pad_w > 0:
            all_flow_vis = all_flow_vis[:, :orig_h, :orig_w, :]

        return (all_flow_vis,)


class MostRecentFileSelector:
    """Selects the most recently modified file in a given directory, or creates a black image if none found."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {
                    "default": folder_paths.get_input_directory(),
                    "tooltip": "Path to search for files. Can be absolute or relative path."
                }),
                "file_types": ("STRING", {
                    "default": "",
                    "tooltip": "Optional comma-separated list of file extensions to filter by (e.g., 'png,jpg'). Leave empty for all files."
                }),
                "create_init_image_if_not_found": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When True, creates a black image if no matching file is found."
                }),
                "init_image_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Width of the black image to create if no file is found."
                }),
                "init_image_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Height of the black image to create if no file is found."
                }),
                "init_image_format": (["png", "jpg"], {
                    "default": "png",
                    "tooltip": "Format to save the black image in if no file is found."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)  # Now we only return the file path
    RETURN_NAMES = ("file_path",)  # Updated return name
    FUNCTION = "select_most_recent_file"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, directory_path, file_types, create_init_image_if_not_found, init_image_width, init_image_height, init_image_format):
        # Returning float("nan") forces this node to always re-execute
        # when the workflow is queued, regardless of input values changing.
        # Include all inputs in IS_CHANGED signature for completeness, even if returning nan.
        return float("nan")

    def select_most_recent_file(self, directory_path, file_types="", create_init_image_if_not_found=False, 
                              init_image_width=512, init_image_height=512, init_image_format="png"):
        """
        Finds the most recently modified file in the specified directory.
        If no file is found and create_init_image_if_not_found is True, creates a black image.

        Args:
            directory_path (str): The path to the directory to search.
            file_types (str): A comma-separated string of file extensions to filter by (e.g., "png,jpg").
                              If empty, all files are considered.
            create_init_image_if_not_found (bool): If True, create a black image if no file is found.
            init_image_width (int): The width of the black image to create.
            init_image_height (int): The height of the black image to create.

        Returns:
            tuple: A tuple containing:
                   - The full path of the most recent file (str), or an empty string if none found.
        """
        # Strip quotation marks from directory path if present
        directory_path = directory_path.strip('"\'')

        def create_and_save_black_image():
            # Create a black image tensor
            black_image = torch.zeros((1, init_image_height, init_image_width, 3), dtype=torch.float32)
            
            # Convert to PIL Image
            # First convert to numpy and then to PIL
            import numpy as np
            from PIL import Image
            
            # Convert tensor to numpy array and scale to 0-255
            img_array = (black_image[0].numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
            
            # Generate unique filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"black_image_{timestamp}.{init_image_format}"
            filepath = os.path.join(directory_path, filename)
            
            # Save the image
            pil_image.save(filepath)
            print(f"Created and saved black image: {filepath}")
            return filepath

        # Check if directory exists
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            print(f"Error: Directory '{directory_path}' does not exist or is not a directory.")
            if create_init_image_if_not_found:
                return (create_and_save_black_image(),)
            return ("",)

        # Parse file types
        extensions = [ext.strip().lower() for ext in file_types.split(",") if ext.strip()]
        # Add dot prefix if not present
        extensions = ["." + ext if not ext.startswith(".") else ext for ext in extensions] if extensions else None


        most_recent_file = None
        most_recent_time = 0

        try:
            # Iterate through entries in the directory
            for entry in os.scandir(directory_path):
                # Check if it's a file
                if entry.is_file():
                    file_path = entry.path
                    # Check file type if extensions are specified
                    if extensions:
                        file_extension = os.path.splitext(file_path)[1].lower()
                        if file_extension not in extensions:
                            continue # Skip if extension doesn't match

                    # Get modification time
                    mod_time = os.path.getmtime(file_path)

                    # Update if this is the most recent file found so far
                    if mod_time > most_recent_time:
                        most_recent_time = mod_time
                        most_recent_file = file_path

        except FileNotFoundError:
             print(f"Directory not found during scan: {directory_path}")
             # If directory not found during scan, return empty path and potentially an init image
             if create_init_image_if_not_found:
                 print(f"Creating black init image: {init_image_width}x{init_image_height}")
                 black_image = torch.zeros((1, init_image_height, init_image_width, 3), dtype=torch.float32)
                 return ("",)
             else:
                 return ("",)
        except Exception as e:
            print(f"An error occurred while scanning directory {directory_path}: {e}")
            # If any other error occurs, return empty path and potentially an init image
            if create_init_image_if_not_found:
                 print(f"Creating black init image: {init_image_width}x{init_image_height}")
                 black_image = torch.zeros((1, init_image_height, init_image_width, 3), dtype=torch.float32)
                 return ("",)
            else:
                 return ("",)


        if most_recent_file:
            print(f"Most recent file in '{directory_path}': {most_recent_file}")
            return (most_recent_file,)
        else:
            # No file found, check if we should create an init image
            if create_init_image_if_not_found:
                if extensions:
                    print(f"No files with extensions {extensions} found in '{directory_path}'. Creating black image.")
                else:
                    print(f"No files found in '{directory_path}'. Creating black image.")
                return (create_and_save_black_image(),)
            else:
                if extensions:
                    print(f"No files with extensions {extensions} found in '{directory_path}'.")
                else:
                    print(f"No files found in '{directory_path}'.")
                return ("",)


class LoadTextLineFromFile:
    """Loads text from a file and returns a specific line by index, random selection, or all lines.
    
    Supports .txt and .csv files, with options for filtering comments, blank lines,
    tag removal, weighting, and text transformations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to a text file (.txt, .csv, etc.) or a directory. "
                               "When a directory is given, a random .txt file is selected from it. "
                               "Surrounding quotes are stripped automatically."
                }),
                "selection_mode": (["random", "index", "all", "sequential"], {
                    "default": "random",
                    "tooltip": (
                        "random: Pick a random line (seeded).\n"
                        "index: Pick a specific line by index number.\n"
                        "all: Return every line joined by the join_delimiter.\n"
                        "sequential: Cycle through lines in order across executions."
                    )
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for random selection. Same seed + same file = same line."
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Line index to retrieve when selection_mode is 'index'. Wraps around if out of range."
                }),
            },
            "optional": {
                "text_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "When provided, this text is used instead of reading from file_path. "
                               "All line selection and filtering still applies. "
                               "Right-click and Convert to Input to connect a wire instead."
                }),
                "encoding": (["utf-8", "utf-8-sig", "latin-1", "ascii", "cp1252"], {
                    "default": "utf-8",
                    "tooltip": "File encoding to use when reading the text file."
                }),
                "ban_tags": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated list of tags/words to remove from the output. "
                               "Surrounding commas and extra spaces are cleaned up."
                }),
                "prefix": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Text to prepend to the output."
                }),
                "suffix": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Text to append to the output."
                }),
                "join_delimiter": ("STRING", {
                    "default": ", ",
                    "tooltip": "Delimiter used to join lines when selection_mode is 'all'."
                }),
                "weight": ("FLOAT", {
                    "default": 0.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "When non-zero, wraps output as (text:weight) for prompt weighting."
                }),
                "skip_comments": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip lines starting with '#' (after stripping whitespace)."
                }),
                "skip_blank_lines": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip empty or whitespace-only lines."
                }),
                "strip_whitespace": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Strip leading/trailing whitespace from each line."
                }),
                "replace_underscores": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Replace underscores with spaces in the output text."
                }),
                "pick_random_file_from_dir": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When True and file_path is a directory, pick a random .txt file from it."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "source_file", "line_count")
    FUNCTION = "load_text_line"
    CATEGORY = "utils"

    _sequential_indices = {}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("selection_mode") == "sequential":
            return float("nan")
        return ""

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Strip quotes, whitespace, and normalize separators."""
        path = path.strip().strip('"\'').strip()
        path = os.path.normpath(path)
        return path

    @staticmethod
    def _read_lines(file_path: str, encoding: str = "utf-8") -> list[str]:
        """Read a file and return raw lines (newlines stripped)."""
        with open(file_path, "r", encoding=encoding, newline="") as f:
            content = f.read()
        lines = []
        for line in io.StringIO(content):
            lines.append(line.rstrip("\n").rstrip("\r"))
        return lines

    @staticmethod
    def _filter_lines(lines: list[str], skip_comments: bool, skip_blank: bool, strip_ws: bool) -> list[str]:
        """Filter and optionally strip lines."""
        result = []
        for line in lines:
            working = line.strip() if strip_ws else line
            if skip_comments and working.lstrip().startswith("#"):
                continue
            if skip_blank and working.strip() == "":
                continue
            result.append(working)
        return result

    @staticmethod
    def _remove_tags(text: str, ban_tags: str) -> str:
        """Remove comma-separated tags from text and clean up leftover delimiters."""
        if not ban_tags.strip():
            return text
        tags = [t.strip() for t in ban_tags.split(",") if t.strip()]
        for tag in tags:
            text = re.sub(
                r",?\s*" + re.escape(tag) + r"\s*,?",
                ",",
                text,
            )
        text = re.sub(r",\s*,", ",", text)
        text = text.strip(", ")
        return text

    def load_text_line(
        self,
        file_path: str = "",
        selection_mode: str = "random",
        seed: int = 0,
        index: int = 0,
        text_override: str = "",
        encoding: str = "utf-8",
        ban_tags: str = "",
        prefix: str = "",
        suffix: str = "",
        join_delimiter: str = ", ",
        weight: float = 0.0,
        skip_comments: bool = True,
        skip_blank_lines: bool = True,
        strip_whitespace: bool = True,
        replace_underscores: bool = False,
        pick_random_file_from_dir: bool = False,
    ):
        source_file = ""

        if text_override is not None and text_override.strip() != "":
            raw_lines = text_override.split("\n")
            source_file = "<text_override>"
        else:
            file_path = self._normalize_path(file_path)

            if not file_path:
                print("[LoadTextLineFromFile] No file path provided.")
                return ("", "", 0)

            if os.path.isdir(file_path):
                if not pick_random_file_from_dir:
                    print(f"[LoadTextLineFromFile] Path is a directory but pick_random_file_from_dir is off: {file_path}")
                    return ("", file_path, 0)

                txt_files = glob_module.glob(os.path.join(file_path, "**", "*.txt"), recursive=True)
                if not txt_files:
                    print(f"[LoadTextLineFromFile] No .txt files found in directory: {file_path}")
                    return ("", file_path, 0)

                rng = random.Random(seed)
                file_path = rng.choice(txt_files)

            if not os.path.isfile(file_path):
                print(f"[LoadTextLineFromFile] File not found: {file_path}")
                return ("", file_path, 0)

            source_file = file_path

            try:
                raw_lines = self._read_lines(file_path, encoding)
            except Exception as e:
                print(f"[LoadTextLineFromFile] Error reading file: {e}")
                return ("", file_path, 0)

        lines = self._filter_lines(raw_lines, skip_comments, skip_blank_lines, strip_whitespace)
        line_count = len(lines)

        if line_count == 0:
            print(f"[LoadTextLineFromFile] No lines remaining after filtering ({source_file}).")
            return ("", source_file, 0)

        if selection_mode == "random":
            rng = random.Random(seed)
            output = rng.choice(lines)
        elif selection_mode == "index":
            output = lines[index % line_count]
        elif selection_mode == "sequential":
            cache_key = source_file or file_path
            idx = self._sequential_indices.get(cache_key, 0)
            output = lines[idx % line_count]
            self._sequential_indices[cache_key] = (idx + 1) % line_count
        else:  # all
            output = join_delimiter.join(lines)

        if replace_underscores:
            output = output.replace("_", " ")

        output = self._remove_tags(output, ban_tags)

        if isinstance(weight, str):
            try:
                weight = float(weight) if weight else 0.0
            except ValueError:
                weight = 0.0
        if weight != 0.0:
            output = f"({output}:{weight})"

        if prefix:
            output = prefix + output
        if suffix:
            output = output + suffix

        return (output, source_file, line_count)


class VideoTransitionBatchMerger:
    """Merge image batches into one video timeline with overlapping or inserted transitions."""

    MAX_BATCHES = 64

    TRANSITIONS = [
        "cut",
        "cross_dissolve",
        "additive_dissolve",
        "linear_wipe_left_to_right",
        "linear_wipe_right_to_left",
        "linear_wipe_top_to_bottom",
        "linear_wipe_bottom_to_top",
        "linear_wipe_diagonal_tl_br",
        "linear_wipe_diagonal_tr_bl",
        "barn_doors_open",
        "barn_doors_close",
        "radial_clock_wipe",
        "iris_circle",
        "iris_diamond",
        "iris_star",
        "push_left",
        "push_right",
        "push_up",
        "push_down",
        "slide_left",
        "slide_right",
        "slide_up",
        "slide_down",
        "dip_to_black",
        "dip_to_white",
        "morph_cut",
        "whip_pan_left",
        "whip_pan_right",
        "whip_pan_up",
        "whip_pan_down",
        "light_leak",
        "zoom",
        "dissolve_zoom",
        "checkerboard",
        "band_slide",
        "glitch",
        "film_burn",
        "spin",
        "page_peel",
        "cube_spin",
    ]
    PAIR_TRANSITIONS = ["same_as_default"] + TRANSITIONS
    NON_OVERLAP_TRANSITIONS = {
        "dip_to_black",
        "dip_to_white",
        "morph_cut",
        "whip_pan_left",
        "whip_pan_right",
        "whip_pan_up",
        "whip_pan_down",
        "light_leak",
    }

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "inputcount": ("INT", {
                "default": 2,
                "min": 2,
                "max": cls.MAX_BATCHES,
                "step": 1,
                "tooltip": "Set the number of image batch inputs, then click Update inputs on the node."
            }),
            "image_batch_1": ("IMAGE", {
                "tooltip": "First clip as a ComfyUI IMAGE batch [frames, height, width, channels]."
            }),
            "image_batch_2": ("IMAGE", {
                "tooltip": "Second clip as a ComfyUI IMAGE batch [frames, height, width, channels]."
            }),
            "fps": ("FLOAT", {
                "default": 24.0,
                "min": 1.0,
                "max": 240.0,
                "step": 0.01,
                "tooltip": "Frames per second for the merged image timeline and audio alignment."
            }),
            "transition_mode": (["auto", "overlap", "non_overlap"], {
                "default": "auto",
                "tooltip": "auto: overlap for A/B roll transitions and non-overlap for dip/camera-artifact transitions."
            }),
            "default_transition": (cls.TRANSITIONS, {
                "default": "cross_dissolve",
                "tooltip": "Transition used when a pair-specific transition is set to same_as_default."
            }),
            "default_transition_frames": ("INT", {
                "default": 12,
                "min": 0,
                "max": 1000,
                "step": 1,
                "tooltip": "Default transition length in frames. Pair-specific values of 0 use this default."
            }),
            "easing": (["linear", "ease_in", "ease_out", "ease_in_out", "smoothstep"], {
                "default": "smoothstep",
                "tooltip": "Progress curve used by opacity, wipe, slide, and inserted transition frames."
            }),
            "size_match": (["strict", "first_batch", "largest_area", "smallest_area"], {
                "default": "first_batch",
                "tooltip": "How to choose the output dimensions when input batches differ."
            }),
            "resize_method": (["stretch", "pad", "crop"], {
                "default": "pad",
                "tooltip": "Resize method used when size_match is not strict and clip dimensions differ."
            }),
            "wipe_softness": ("FLOAT", {
                "default": 0.08,
                "min": 0.0,
                "max": 0.5,
                "step": 0.01,
                "tooltip": "Soft edge width for wipe and iris transitions as a fraction of the frame."
            }),
            "effect_intensity": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 3.0,
                "step": 0.05,
                "tooltip": "Strength for bloom, light leak, and whip-pan blur effects."
            }),
        }

        optional = {
            "audio_1": ("AUDIO,VHS_AUDIO", {
                "tooltip": "Optional audio paired with image_batch_1. Accepts ComfyUI AUDIO or legacy VHS_AUDIO."
            }),
            "audio_2": ("AUDIO,VHS_AUDIO", {
                "tooltip": "Optional audio paired with image_batch_2. Accepts ComfyUI AUDIO or legacy VHS_AUDIO."
            }),
            "transition_plan": ("STRING", {
                "default": "",
                "multiline": True,
                "tooltip": (
                    "Optional per-pair overrides, one per line. Examples:\n"
                    "1:cross_dissolve:12\n"
                    "2:dip_to_white:8\n"
                    "The web UI manages this automatically when using dynamic transition controls."
                )
            })
        }

        for index in range(3, cls.MAX_BATCHES + 1):
            optional[f"image_batch_{index}"] = ("IMAGE", {
                "tooltip": (
                    f"Optional API-safe clip input for image_batch_{index}. "
                    "Set inputcount high enough for this input to be used."
                )
            })
            optional[f"audio_{index}"] = ("AUDIO,VHS_AUDIO", {
                "tooltip": f"Optional audio paired with image_batch_{index}."
            })

        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count", "timeline_info")
    FUNCTION = "merge_transitions"
    CATEGORY = "Image/Animation"

    @staticmethod
    def _validate_batch(images, name):
        if images is None:
            raise ValueError(f"{name} is required.")
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor IMAGE batch.")
        if images.ndim != 4:
            raise ValueError(f"{name} must have shape [frames, height, width, channels], got {tuple(images.shape)}.")
        if images.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one frame.")
        if images.shape[-1] < 1:
            raise ValueError(f"{name} must contain at least one channel.")
        return images

    @staticmethod
    def _apply_easing(t, easing):
        t = t.clamp(0.0, 1.0)
        if easing == "linear":
            return t
        if easing == "ease_in":
            return t * t
        if easing == "ease_out":
            inv = 1.0 - t
            return 1.0 - inv * inv
        if easing == "ease_in_out":
            return torch.where(t < 0.5, 2.0 * t * t, 1.0 - torch.pow(-2.0 * t + 2.0, 2.0) / 2.0)
        return t * t * (3.0 - 2.0 * t)

    def _progress(self, frame_count, device, dtype, easing):
        if frame_count <= 0:
            return torch.empty((0, 1, 1, 1), device=device, dtype=dtype)
        values = torch.linspace(0.0, 1.0, frame_count + 2, device=device, dtype=dtype)[1:-1]
        return self._apply_easing(values.view(frame_count, 1, 1, 1), easing)

    @staticmethod
    def _mix(a, b, alpha):
        return a * (1.0 - alpha) + b * alpha

    @staticmethod
    def _normalize_transition_name(transition):
        normalized = str(transition).strip().lower().replace("-", "_")
        aliases = {
            "cross_dissolve": "cross_dissolve",
            "crossfade": "cross_dissolve",
            "dip_black": "dip_to_black",
            "fade_black": "dip_to_black",
            "fade_to_black": "dip_to_black",
            "dip_white": "dip_to_white",
            "fade_white": "dip_to_white",
            "fade_through_white": "dip_to_white",
            "iris_wipe": "iris_circle",
            "clock_wipe": "radial_clock_wipe",
            "morph": "morph_cut",
            "film_burn": "film_burn",
            "page_peel": "page_peel",
            "cube_spin": "cube_spin",
            "dissolve_zoom": "dissolve_zoom",
            "band_slide": "band_slide",
        }
        return aliases.get(normalized, normalized)

    @staticmethod
    def _target_size(batches, size_match):
        sizes = [(int(batch.shape[1]), int(batch.shape[2])) for batch in batches]
        if size_match == "strict":
            if len(set(sizes)) != 1:
                raise ValueError(f"All image batches must have the same height/width in strict mode, got {sizes}.")
            return sizes[0]
        if size_match == "largest_area":
            return max(sizes, key=lambda size: size[0] * size[1])
        if size_match == "smallest_area":
            return min(sizes, key=lambda size: size[0] * size[1])
        return sizes[0]

    @staticmethod
    def _resize_batch(images, target_h, target_w, resize_method):
        if int(images.shape[1]) == target_h and int(images.shape[2]) == target_w:
            return images

        n, h, w, c = images.shape
        nchw = images.permute(0, 3, 1, 2)

        if resize_method == "stretch":
            resized = torch_nn_func.interpolate(nchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
            return resized.permute(0, 2, 3, 1).contiguous()

        if resize_method == "crop":
            scale = max(target_w / float(w), target_h / float(h))
            new_w = max(target_w, int(np.ceil(w * scale)))
            new_h = max(target_h, int(np.ceil(h * scale)))
            resized = torch_nn_func.interpolate(nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
            top = max(0, (new_h - target_h) // 2)
            left = max(0, (new_w - target_w) // 2)
            resized = resized[:, :, top:top + target_h, left:left + target_w]
            return resized.permute(0, 2, 3, 1).contiguous()

        scale = min(target_w / float(w), target_h / float(h))
        new_w = max(1, min(target_w, int(round(w * scale))))
        new_h = max(1, min(target_h, int(round(h * scale))))
        resized = torch_nn_func.interpolate(nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        resized = torch_nn_func.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        return resized.permute(0, 2, 3, 1).contiguous()

    def _prepare_batches(self, batches, size_match, resize_method):
        channels = int(batches[0].shape[-1])
        target_h, target_w = self._target_size(batches, size_match)
        device = batches[0].device
        dtype = batches[0].dtype

        prepared = []
        for index, batch in enumerate(batches, start=1):
            if int(batch.shape[-1]) != channels:
                raise ValueError(
                    f"All image batches must have the same channel count. "
                    f"image_batch_1 has {channels}, image_batch_{index} has {int(batch.shape[-1])}."
                )
            batch = batch.to(device=device, dtype=dtype)
            if size_match != "strict":
                batch = self._resize_batch(batch, target_h, target_w, resize_method)
            prepared.append(batch.contiguous())
        return prepared

    @staticmethod
    def _frame_to_sample(frame_count, fps, sample_rate):
        return int(round(float(frame_count) / float(fps) * int(sample_rate)))

    @staticmethod
    def _resize_audio_samples(waveform, target_samples):
        target_samples = max(0, int(target_samples))
        if waveform.shape[-1] == target_samples:
            return waveform
        if target_samples == 0:
            return waveform[..., :0]
        if waveform.shape[-1] == 0:
            shape = list(waveform.shape)
            shape[-1] = target_samples
            return torch.zeros(shape, dtype=waveform.dtype, device=waveform.device)
        return torch_nn_func.interpolate(waveform, size=target_samples, mode="linear", align_corners=False)

    def _coerce_audio_input(self, audio, name="audio"):
        if audio is None:
            return audio
        if isinstance(audio, Mapping):
            if "waveform" in audio and "sample_rate" in audio:
                return {"waveform": audio["waveform"], "sample_rate": audio["sample_rate"]}
            return audio
        if callable(audio):
            return self._decode_vhs_audio(audio, name)
        raise ValueError(f"{name} must be ComfyUI AUDIO, dict-like AUDIO, or legacy VHS_AUDIO.")

    def _decode_vhs_audio(self, vhs_audio, name):
        try:
            payload = vhs_audio()
        except Exception as error:
            raise ValueError(f"{name} VHS_AUDIO callable failed: {error}") from error

        if not payload:
            raise ValueError(f"{name} VHS_AUDIO input is empty.")

        try:
            import subprocess
            from videohelpersuite.utils import ENCODE_ARGS, ffmpeg_path

            result = subprocess.run(
                [ffmpeg_path, "-i", "-", "-f", "f32le", "-"],
                input=payload,
                capture_output=True,
                check=True,
            )
            waveform = torch.frombuffer(bytearray(result.stdout), dtype=torch.float32)
            stderr = result.stderr.decode(*ENCODE_ARGS)
        except Exception as ffmpeg_error:
            try:
                import av

                frames = []
                with av.open(io.BytesIO(payload), mode="r") as container:
                    if not container.streams.audio:
                        raise ValueError("VHS_AUDIO payload has no audio stream.")
                    stream = container.streams.audio[0]
                    sample_rate = int(stream.rate or 44100)
                    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sample_rate)
                    for frame in container.decode(stream):
                        resampled_frames = resampler.resample(frame) or []
                        for resampled in resampled_frames:
                            frames.append(torch.from_numpy(resampled.to_ndarray()))
                    flushed_frames = resampler.resample(None) or []
                    for resampled in flushed_frames:
                        frames.append(torch.from_numpy(resampled.to_ndarray()))

                if not frames:
                    raise ValueError("VHS_AUDIO payload decoded to zero audio frames.")
                waveform = torch.cat(frames, dim=1).unsqueeze(0).to(dtype=torch.float32)
                return {"waveform": waveform, "sample_rate": sample_rate}
            except Exception as pyav_error:
                raise ValueError(
                    f"{name} VHS_AUDIO could not be decoded. "
                    f"ffmpeg error: {ffmpeg_error}; PyAV error: {pyav_error}"
                ) from pyav_error

        match = re.search(r", (\d+) Hz, (\w+), ", stderr)
        if match:
            sample_rate = int(match.group(1))
            channel_layout = match.group(2)
            channels = {"mono": 1, "stereo": 2}.get(channel_layout)
            if channels is None:
                raise ValueError(f"{name} VHS_AUDIO channel layout is not supported: {channel_layout}.")
        else:
            sample_rate = 44100
            channels = 2

        if waveform.numel() % channels != 0:
            waveform = waveform[: waveform.numel() - (waveform.numel() % channels)]
        waveform = waveform.reshape((-1, channels)).transpose(0, 1).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": sample_rate}

    def _normalize_audio_waveform(self, audio, sample_rate, target_channels=2, target_samples=None):
        audio = self._coerce_audio_input(audio)
        if audio is None:
            if target_samples is None:
                target_samples = 0
            return torch.zeros((1, target_channels, target_samples), dtype=torch.float32)

        if not isinstance(audio, Mapping) or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("Audio inputs must use ComfyUI AUDIO, dict-like AUDIO, or legacy VHS_AUDIO.")

        waveform = audio["waveform"]
        if not isinstance(waveform, torch.Tensor):
            raise ValueError("Audio waveform must be a torch.Tensor.")
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError(f"Audio waveform must have shape [batch, channels, samples], got {tuple(waveform.shape)}.")
        if waveform.shape[0] > 1:
            print("[VideoTransitionBatchMerger] Audio batch dimension > 1; using the first audio item.")
            waveform = waveform[:1]

        waveform = waveform.to(dtype=torch.float32)
        input_sample_rate = int(audio["sample_rate"])
        if input_sample_rate <= 0:
            raise ValueError("Audio sample_rate must be greater than 0.")

        if waveform.shape[1] == 1 and target_channels == 2:
            waveform = waveform.repeat(1, 2, 1)
        elif waveform.shape[1] > target_channels:
            waveform = waveform[:, :target_channels, :]
        elif waveform.shape[1] < target_channels:
            repeat_count = int(np.ceil(target_channels / waveform.shape[1]))
            waveform = waveform.repeat(1, repeat_count, 1)[:, :target_channels, :]

        if input_sample_rate != sample_rate:
            target_resampled = self._frame_to_sample(waveform.shape[-1], input_sample_rate, sample_rate)
            waveform = self._resize_audio_samples(waveform, target_resampled)

        if target_samples is not None:
            waveform = self._resize_audio_samples(waveform, target_samples)

        return waveform

    def _choose_audio_sample_rate(self, audios, default_sample_rate=44100):
        for audio in audios:
            if audio is not None and isinstance(audio, dict) and "sample_rate" in audio:
                sample_rate = int(audio["sample_rate"])
                if sample_rate > 0:
                    return sample_rate
        return int(default_sample_rate)

    def _prepare_audio_clips(self, audios, frame_counts, fps, sample_rate):
        clips = []
        target_channels = 2
        for audio, frame_count in zip(audios, frame_counts):
            target_samples = self._frame_to_sample(frame_count, fps, sample_rate)
            if audio is None:
                clips.append(torch.zeros((1, target_channels, target_samples), dtype=torch.float32))
            else:
                clips.append(self._normalize_audio_waveform(audio, sample_rate, target_channels, target_samples))
        return clips

    def _audio_slice_for_frames(self, waveform, start_frame, end_frame, fps, sample_rate):
        start_sample = self._frame_to_sample(start_frame, fps, sample_rate)
        end_sample = self._frame_to_sample(end_frame, fps, sample_rate)
        return waveform[..., start_sample:end_sample]

    def _silence_for_frames(self, frame_count, reference_audio, fps, sample_rate):
        samples = self._frame_to_sample(frame_count, fps, sample_rate)
        return torch.zeros(
            (1, reference_audio.shape[1], samples),
            dtype=reference_audio.dtype,
            device=reference_audio.device,
        )

    def _fit_audio_to_frames(self, waveform, frame_count, fps, sample_rate):
        target_samples = self._frame_to_sample(frame_count, fps, sample_rate)
        return self._resize_audio_samples(waveform, target_samples)

    def _mix_overlap_audio(self, outgoing, incoming, easing):
        target_samples = max(outgoing.shape[-1], incoming.shape[-1])
        outgoing = self._resize_audio_samples(outgoing, target_samples)
        incoming = self._resize_audio_samples(incoming, target_samples)
        if target_samples == 0:
            return outgoing

        progress = torch.linspace(
            0.0,
            1.0,
            target_samples + 2,
            device=outgoing.device,
            dtype=outgoing.dtype,
        )[1:-1].view(1, 1, target_samples)
        progress = self._apply_easing(progress, easing)
        mixed = outgoing * (1.0 - progress) + incoming * progress
        return mixed.clamp(-1.0, 1.0)

    @staticmethod
    def _soft_reveal_less(field, threshold, softness):
        if softness <= 0.0:
            return (field <= threshold).to(field.dtype)
        return ((threshold - field) / softness + 0.5).clamp(0.0, 1.0)

    @staticmethod
    def _soft_reveal_greater(field, threshold, softness):
        if softness <= 0.0:
            return (field >= threshold).to(field.dtype)
        return ((field - threshold) / softness + 0.5).clamp(0.0, 1.0)

    def _render_wipe(self, a, b, transition, progress, wipe_softness):
        n, h, w, _ = a.shape
        device = a.device
        dtype = a.dtype
        xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype).view(1, 1, w, 1)
        ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype).view(1, h, 1, 1)
        p = progress
        softness = max(0.0, float(wipe_softness))

        if transition == "linear_wipe_left_to_right":
            field = xs
            mask = self._soft_reveal_less(field, p, softness)
        elif transition == "linear_wipe_right_to_left":
            field = 1.0 - xs
            mask = self._soft_reveal_less(field, p, softness)
        elif transition == "linear_wipe_top_to_bottom":
            field = ys
            mask = self._soft_reveal_less(field, p, softness)
        elif transition == "linear_wipe_bottom_to_top":
            field = 1.0 - ys
            mask = self._soft_reveal_less(field, p, softness)
        elif transition == "linear_wipe_diagonal_tl_br":
            field = (xs + ys) * 0.5
            mask = self._soft_reveal_less(field, p, softness)
        elif transition == "linear_wipe_diagonal_tr_bl":
            field = ((1.0 - xs) + ys) * 0.5
            mask = self._soft_reveal_less(field, p, softness)
        elif transition == "barn_doors_open":
            field = torch.abs(xs - 0.5) * 2.0
            mask = self._soft_reveal_less(field, p, softness)
        elif transition == "barn_doors_close":
            field = torch.abs(xs - 0.5) * 2.0
            mask = self._soft_reveal_greater(field, 1.0 - p, softness)
        elif transition == "radial_clock_wipe":
            angle = torch.atan2(ys - 0.5, xs - 0.5) + (np.pi / 2.0)
            field = torch.remainder(angle, 2.0 * np.pi) / (2.0 * np.pi)
            mask = self._soft_reveal_less(field, p, max(softness * 0.5, 0.0001))
        elif transition == "iris_circle":
            field = torch.sqrt(torch.pow(xs - 0.5, 2.0) + torch.pow(ys - 0.5, 2.0)) / np.sqrt(0.5)
            mask = self._soft_reveal_less(field, p, softness)
        elif transition == "iris_diamond":
            field = torch.abs(xs - 0.5) + torch.abs(ys - 0.5)
            mask = self._soft_reveal_less(field, p, softness)
        else:
            dx = xs - 0.5
            dy = ys - 0.5
            radius = torch.sqrt(dx * dx + dy * dy) / np.sqrt(0.5)
            angle = torch.atan2(dy, dx)
            star_edge = 0.58 + 0.30 * torch.cos(5.0 * angle)
            field = radius / star_edge.clamp(min=0.08)
            mask = self._soft_reveal_less(field, p * 1.35, softness)

        if mask.shape[0] == 1 and n > 1:
            mask = mask.expand(n, -1, -1, -1)
        return a * (1.0 - mask) + b * mask

    @staticmethod
    def _shift_frame(frame, dx, dy):
        h, w, _ = frame.shape
        dx = int(dx)
        dy = int(dy)
        out = torch.zeros_like(frame)

        src_x0 = max(0, -dx)
        src_x1 = min(w, w - dx)
        dst_x0 = max(0, dx)
        dst_x1 = min(w, w + dx)

        src_y0 = max(0, -dy)
        src_y1 = min(h, h - dy)
        dst_y0 = max(0, dy)
        dst_y1 = min(h, h + dy)

        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return out

        out[dst_y0:dst_y1, dst_x0:dst_x1, :] = frame[src_y0:src_y1, src_x0:src_x1, :]
        return out

    @staticmethod
    def _offsets(direction, progress_value, height, width):
        if direction == "left":
            return -round(progress_value * width), 0, round((1.0 - progress_value) * width), 0
        if direction == "right":
            return round(progress_value * width), 0, -round((1.0 - progress_value) * width), 0
        if direction == "up":
            return 0, -round(progress_value * height), 0, round((1.0 - progress_value) * height)
        return 0, round(progress_value * height), 0, -round((1.0 - progress_value) * height)

    def _render_slide_push(self, a, b, transition, progress):
        n, h, w, c = a.shape
        direction = transition.split("_", 1)[1]
        is_push = transition.startswith("push_")
        frames = []
        base_mask = torch.ones((h, w, 1), device=a.device, dtype=a.dtype)

        for index in range(n):
            p = float(progress[index].reshape(-1)[0].item())
            ax, ay, bx, by = self._offsets(direction, p, h, w)
            shifted_b = self._shift_frame(b[index], bx, by)
            shifted_b_mask = self._shift_frame(base_mask, bx, by)

            if is_push:
                shifted_a = self._shift_frame(a[index], ax, ay)
                frame = shifted_a * (1.0 - shifted_b_mask) + shifted_b
            else:
                frame = a[index] * (1.0 - shifted_b_mask) + shifted_b
            frames.append(frame.clamp(0.0, 1.0))

        return torch.stack(frames, dim=0)

    def _render_additive(self, a, b, progress, effect_intensity):
        base = self._mix(a, b, progress)
        peak = 4.0 * progress * (1.0 - progress)
        effect = max(0.0, float(effect_intensity))
        out = base + torch.clamp(a + b, 0.0, 1.0) * peak * 0.35 * effect + peak * 0.25 * effect
        return out.clamp(0.0, 1.0)

    def _render_dip(self, a, b, transition, progress):
        color_value = 1.0 if transition == "dip_to_white" else 0.0
        color = torch.full_like(a, color_value)
        first_half = progress < 0.5
        fade_out = self._mix(a, color, progress * 2.0)
        fade_in = self._mix(color, b, (progress - 0.5) * 2.0)
        return torch.where(first_half, fade_out, fade_in).clamp(0.0, 1.0)

    def _render_morph_cut(self, a, b, progress, effect_intensity):
        smooth = self._apply_easing(progress, "smoothstep")
        base = self._mix(a, b, smooth)
        detail_soften = 4.0 * smooth * (1.0 - smooth) * min(max(float(effect_intensity), 0.0), 3.0) * 0.04
        return (base + (b - a).abs() * detail_soften).clamp(0.0, 1.0)

    def _directional_blur_frame(self, frame, axis, radius):
        radius = int(max(0, min(radius, 64)))
        if radius == 0:
            return frame
        sample_count = min(radius * 2 + 1, 17)
        offsets = sorted({int(round(v)) for v in np.linspace(-radius, radius, sample_count)})
        acc = torch.zeros_like(frame)
        for offset in offsets:
            dx = offset if axis == "horizontal" else 0
            dy = offset if axis == "vertical" else 0
            acc = acc + self._shift_frame(frame, dx, dy)
        return acc / float(len(offsets))

    def _render_whip_pan(self, a, b, transition, progress, effect_intensity):
        direction = transition.replace("whip_pan_", "")
        pushed = self._render_slide_push(a, b, f"push_{direction}", progress)
        axis = "horizontal" if direction in {"left", "right"} else "vertical"
        frames = []
        for index in range(pushed.shape[0]):
            p = float(progress[index].reshape(-1)[0].item())
            peak = max(0.0, float(np.sin(np.pi * p)))
            radius = int(round((4.0 + 20.0 * max(0.0, float(effect_intensity))) * peak))
            frames.append(self._directional_blur_frame(pushed[index], axis, radius).clamp(0.0, 1.0))
        return torch.stack(frames, dim=0)

    def _render_light_leak(self, a, b, progress, effect_intensity):
        n, h, w, c = a.shape
        device = a.device
        dtype = a.dtype
        base = self._mix(a, b, progress)
        xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype).view(1, 1, w, 1)
        ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype).view(1, h, 1, 1)
        peak = torch.sin(progress * np.pi).clamp(0.0, 1.0)
        center_x = 0.15 + progress * 0.70
        center_y = 0.50 + 0.18 * torch.sin(progress * np.pi * 2.0)
        dist = torch.pow(xs - center_x, 2.0) + torch.pow(ys - center_y, 2.0)
        flare = (torch.exp(-dist / 0.035) + 0.35) * peak * max(0.0, float(effect_intensity))
        color = torch.ones((1, 1, 1, c), device=device, dtype=dtype)
        if c >= 3:
            color[..., 0] = 1.0
            color[..., 1] = 0.72
            color[..., 2] = 0.28
        white_peak = torch.pow(peak, 4.0) * 0.40 * min(max(float(effect_intensity), 0.0), 3.0)
        return (base + flare * color + white_peak).clamp(0.0, 1.0)

    @staticmethod
    def _uv_fields(images):
        n, h, w, _ = images.shape
        device = images.device
        dtype = images.dtype
        xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype).view(1, 1, w, 1)
        ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype).view(1, h, 1, 1)
        return xs.expand(n, h, w, 1), ys.expand(n, h, w, 1)

    @staticmethod
    def _pseudo_noise(field):
        return torch.remainder(torch.sin(field) * 43758.5453123, 1.0)

    def _sample_batch(self, images, sample_x, sample_y):
        n, h, w, _ = images.shape
        if sample_x.shape[0] == 1 and n > 1:
            sample_x = sample_x.expand(n, -1, -1, -1)
        if sample_y.shape[0] == 1 and n > 1:
            sample_y = sample_y.expand(n, -1, -1, -1)
        grid = torch.cat([sample_x * 2.0 - 1.0, sample_y * 2.0 - 1.0], dim=-1)
        sampled = torch_nn_func.grid_sample(
            images.permute(0, 3, 1, 2),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _bounds_mask(sample_x, sample_y, n):
        mask = ((sample_x >= 0.0) & (sample_x <= 1.0) & (sample_y >= 0.0) & (sample_y <= 1.0)).to(sample_x.dtype)
        if mask.shape[0] == 1 and n > 1:
            mask = mask.expand(n, -1, -1, -1)
        return mask

    def _render_zoom(self, a, b, progress, effect_intensity):
        xs, ys = self._uv_fields(a)
        effect = min(max(float(effect_intensity), 0.0), 3.0)
        peak = torch.sin(progress * np.pi).clamp(0.0, 1.0)
        scale_a = 1.0 + 0.10 * effect * progress
        scale_b = 0.55 + 0.45 * progress
        ax = (xs - 0.5) / scale_a + 0.5
        ay = (ys - 0.5) / scale_a + 0.5
        bx = (xs - 0.5) / scale_b + 0.5
        by = (ys - 0.5) / scale_b + 0.5
        a_zoom = self._sample_batch(a, ax, ay)
        b_zoom = self._sample_batch(b, bx, by)
        out = self._mix(a_zoom, b_zoom, progress)
        return (out + peak * 0.03 * effect).clamp(0.0, 1.0)

    def _render_dissolve_zoom(self, a, b, progress, effect_intensity):
        xs, ys = self._uv_fields(a)
        effect = min(max(float(effect_intensity), 0.0), 3.0)
        scale_a = 1.0 + 0.22 * effect * progress
        scale_b = 1.12 - 0.12 * progress
        a_zoom = self._sample_batch(a, (xs - 0.5) / scale_a + 0.5, (ys - 0.5) / scale_a + 0.5)
        b_zoom = self._sample_batch(b, (xs - 0.5) / scale_b + 0.5, (ys - 0.5) / scale_b + 0.5)
        noise = self._pseudo_noise(xs * 127.1 + ys * 311.7)
        alpha = (progress * (1.0 + 0.20 * effect) + (noise - 0.5) * 0.30).clamp(0.0, 1.0)
        return self._mix(a_zoom, b_zoom, alpha).clamp(0.0, 1.0)

    def _render_checkerboard(self, a, b, progress, effect_intensity):
        n, h, w, _ = a.shape
        xs, ys = self._uv_fields(a)
        effect = min(max(float(effect_intensity), 0.0), 3.0)
        rows = max(4, int(round(6 + effect * 4)))
        cols = max(4, int(round(rows * w / max(h, 1))))
        tile_x = torch.floor(xs * cols)
        tile_y = torch.floor(ys * rows)
        phase = torch.remainder(tile_x + tile_y, 2.0) * 0.42
        mask = ((progress - phase) / 0.58).clamp(0.0, 1.0)
        mask = self._apply_easing(mask, "smoothstep")
        return self._mix(a, b, mask).clamp(0.0, 1.0)

    def _render_band_slide(self, a, b, progress, effect_intensity):
        n, _, _, _ = a.shape
        xs, ys = self._uv_fields(a)
        effect = min(max(float(effect_intensity), 0.0), 3.0)
        bands = max(4, int(round(8 + effect * 5)))
        band = torch.floor(ys * bands)
        direction = torch.where(torch.remainder(band, 2.0) == 0.0, 1.0, -1.0)
        offset = direction * (1.0 - progress)
        sample_x = xs - offset
        sample_y = ys
        slid_b = self._sample_batch(b, sample_x, sample_y)
        mask = self._bounds_mask(sample_x, sample_y, n)
        shadow = (1.0 - torch.abs(sample_x.clamp(0.0, 1.0) - xs).clamp(0.0, 1.0)) * 0.08
        out = a * (1.0 - mask) + slid_b * mask
        return (out - shadow * mask * (1.0 - progress)).clamp(0.0, 1.0)

    def _render_glitch(self, a, b, progress, effect_intensity):
        n, h, w, c = a.shape
        xs, ys = self._uv_fields(a)
        effect = min(max(float(effect_intensity), 0.0), 3.0)
        peak = torch.sin(progress * np.pi).clamp(0.0, 1.0)
        band_count = max(12, int(round(18 + effect * 8)))
        bands = torch.floor(ys * band_count)
        band_noise = self._pseudo_noise(bands * 19.19 + progress * 113.7)
        offset = (band_noise - 0.5) * peak * (0.12 + 0.05 * effect)
        a_shifted = self._sample_batch(a, xs + offset * progress, ys)
        b_shifted = self._sample_batch(b, xs - offset * (1.0 - progress), ys)
        alpha = (progress + (band_noise - 0.5) * peak * 0.45).clamp(0.0, 1.0)
        base = self._mix(a_shifted, b_shifted, alpha)
        if c >= 3:
            split = peak * (0.006 + 0.010 * effect)
            red = self._sample_batch(base, xs + split, ys)[..., 0:1]
            green = base[..., 1:2]
            blue = self._sample_batch(base, xs - split, ys)[..., 2:3]
            if c > 3:
                base = torch.cat([red, green, blue, base[..., 3:]], dim=-1)
            else:
                base = torch.cat([red, green, blue], dim=-1)
        tear = (band_noise > (0.82 - 0.04 * effect)).to(base.dtype) * peak
        return (base + tear * 0.08).clamp(0.0, 1.0)

    def _render_film_burn(self, a, b, progress, effect_intensity):
        _, _, _, c = a.shape
        xs, ys = self._uv_fields(a)
        effect = min(max(float(effect_intensity), 0.0), 3.0)
        edge = progress * 1.35 - 0.18
        field = xs + 0.12 * torch.sin(ys * 17.0 + progress * 6.0)
        reveal = ((edge - field) / 0.16 + 0.5).clamp(0.0, 1.0)
        reveal = self._apply_easing(reveal, "smoothstep")
        base = self._mix(a, b, reveal)
        peak = torch.sin(progress * np.pi).clamp(0.0, 1.0)
        hot_edge = torch.exp(-torch.pow(field - edge, 2.0) / 0.010) * peak
        burn_color = torch.ones((1, 1, 1, c), device=a.device, dtype=a.dtype)
        if c >= 3:
            burn_color[..., 0] = 1.0
            burn_color[..., 1] = 0.45
            burn_color[..., 2] = 0.08
        white = torch.pow(peak, 5.0) * 0.55
        return (base + hot_edge * burn_color * (0.75 + 0.35 * effect) + white).clamp(0.0, 1.0)

    def _rotated_sample(self, images, angle, scale):
        xs, ys = self._uv_fields(images)
        dx = xs - 0.5
        dy = ys - 0.5
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        safe_scale = scale.clamp(min=0.001)
        sample_x = (dx * cos_a + dy * sin_a) / safe_scale + 0.5
        sample_y = (-dx * sin_a + dy * cos_a) / safe_scale + 0.5
        return self._sample_batch(images, sample_x, sample_y)

    def _render_spin(self, a, b, progress, effect_intensity):
        effect = min(max(float(effect_intensity), 0.0), 3.0)
        peak = torch.sin(progress * np.pi).clamp(0.0, 1.0)
        angle = progress * np.pi * (1.0 + 0.25 * effect)
        scale = 1.0 + peak * 0.18
        a_spin = self._rotated_sample(a, angle, scale)
        b_spin = self._rotated_sample(b, angle - np.pi * (1.0 + 0.25 * effect), scale)
        return self._mix(a_spin, b_spin, progress).clamp(0.0, 1.0)

    def _render_page_peel(self, a, b, progress, effect_intensity):
        n, _, _, _ = a.shape
        xs, ys = self._uv_fields(a)
        effect = min(max(float(effect_intensity), 0.0), 3.0)
        edge = 1.08 - progress * 1.30
        field = xs + (1.0 - ys) * 0.22
        reveal = ((field - edge) / 0.035 + 0.5).clamp(0.0, 1.0)
        base = self._mix(a, b, reveal)
        curl_width = 0.10 + 0.025 * effect
        curl_mask = (1.0 - torch.abs(field - edge) / curl_width).clamp(0.0, 1.0)
        mirrored_x = (2.0 * edge - xs).clamp(-0.25, 1.25)
        curled = self._sample_batch(a, mirrored_x, ys)
        curl_bounds = self._bounds_mask(mirrored_x, ys, n)
        highlight = curl_mask * (0.18 + 0.08 * effect)
        shadow = curl_mask * (1.0 - reveal) * 0.22
        out = self._mix(base, curled + highlight, curl_mask * curl_bounds * (1.0 - reveal * 0.7))
        return (out - shadow).clamp(0.0, 1.0)

    def _render_cube_spin(self, a, b, progress, effect_intensity):
        n, _, _, _ = a.shape
        xs, ys = self._uv_fields(a)
        angle = progress * (np.pi / 2.0)
        scale_a = torch.cos(angle).clamp(min=0.001)
        scale_b = torch.sin(angle).clamp(min=0.001)
        center_a = 0.5 - 0.5 * torch.sin(angle)
        center_b = 1.0 - 0.5 * torch.sin(angle)
        ax = (xs - center_a) / scale_a + 0.5
        bx = (xs - center_b) / scale_b + 0.5
        mask_a = (torch.abs(xs - center_a) <= scale_a * 0.5).to(a.dtype)
        mask_b = (torch.abs(xs - center_b) <= scale_b * 0.5).to(a.dtype)
        mask_a = mask_a * self._bounds_mask(ax, ys, n)
        mask_b = mask_b * self._bounds_mask(bx, ys, n)
        face_a = self._sample_batch(a, ax, ys) * (0.42 + 0.58 * scale_a)
        face_b = self._sample_batch(b, bx, ys) * (0.42 + 0.58 * scale_b)
        out = face_a * mask_a * (1.0 - mask_b) + face_b * mask_b
        gap = 1.0 - (mask_a + mask_b).clamp(0.0, 1.0)
        return (out + gap * 0.015).clamp(0.0, 1.0)

    def _render_transition(self, a, b, transition, progress, wipe_softness, effect_intensity):
        if transition == "cross_dissolve":
            return self._mix(a, b, progress)
        if transition == "additive_dissolve":
            return self._render_additive(a, b, progress, effect_intensity)
        if transition.startswith("linear_wipe_") or transition.startswith("barn_doors_") or transition.startswith("iris_") or transition == "radial_clock_wipe":
            return self._render_wipe(a, b, transition, progress, wipe_softness)
        if transition.startswith("push_") or transition.startswith("slide_"):
            return self._render_slide_push(a, b, transition, progress)
        if transition in {"dip_to_black", "dip_to_white"}:
            return self._render_dip(a, b, transition, progress)
        if transition == "morph_cut":
            return self._render_morph_cut(a, b, progress, effect_intensity)
        if transition.startswith("whip_pan_"):
            return self._render_whip_pan(a, b, transition, progress, effect_intensity)
        if transition == "light_leak":
            return self._render_light_leak(a, b, progress, effect_intensity)
        if transition == "zoom":
            return self._render_zoom(a, b, progress, effect_intensity)
        if transition == "dissolve_zoom":
            return self._render_dissolve_zoom(a, b, progress, effect_intensity)
        if transition == "checkerboard":
            return self._render_checkerboard(a, b, progress, effect_intensity)
        if transition == "band_slide":
            return self._render_band_slide(a, b, progress, effect_intensity)
        if transition == "glitch":
            return self._render_glitch(a, b, progress, effect_intensity)
        if transition == "film_burn":
            return self._render_film_burn(a, b, progress, effect_intensity)
        if transition == "spin":
            return self._render_spin(a, b, progress, effect_intensity)
        if transition == "page_peel":
            return self._render_page_peel(a, b, progress, effect_intensity)
        if transition == "cube_spin":
            return self._render_cube_spin(a, b, progress, effect_intensity)
        return b

    def _mode_for_transition(self, transition, transition_mode):
        if transition_mode in {"overlap", "non_overlap"}:
            return transition_mode
        return "non_overlap" if transition in self.NON_OVERLAP_TRANSITIONS else "overlap"

    def _parse_transition_plan(self, transition_plan):
        plan = {}
        if not transition_plan or not str(transition_plan).strip():
            return plan

        entries = re.split(r"[\n;]+", str(transition_plan))
        for raw_entry in entries:
            entry = raw_entry.strip()
            if not entry or entry.startswith("#"):
                continue

            parts = [part.strip() for part in re.split(r"[:=,\s]+", entry) if part.strip()]
            if len(parts) < 2:
                raise ValueError(f"Invalid transition_plan entry: '{entry}'. Use 'pair:transition:frames'.")

            try:
                pair_index = int(parts[0])
            except ValueError as exc:
                raise ValueError(f"Invalid transition_plan pair index in '{entry}'.") from exc

            transition = self._normalize_transition_name(parts[1])
            if transition not in self.PAIR_TRANSITIONS:
                raise ValueError(
                    f"Invalid transition '{transition}' in transition_plan entry '{entry}'. "
                    f"Valid transitions: {', '.join(self.PAIR_TRANSITIONS)}"
                )

            frames = None
            if len(parts) >= 3:
                try:
                    frames = max(0, int(parts[2]))
                except ValueError as exc:
                    raise ValueError(f"Invalid frame count in transition_plan entry '{entry}'.") from exc

            plan[pair_index] = (transition, frames)

        return plan

    def _merge_pair(self, current, next_batch, transition, requested_frames, transition_mode, easing, wipe_softness, effect_intensity):
        if transition == "cut" or requested_frames <= 0:
            return torch.cat([current, next_batch], dim=0), "cut", 0

        mode = self._mode_for_transition(transition, transition_mode)

        if mode == "overlap":
            frames = min(int(requested_frames), int(current.shape[0]), int(next_batch.shape[0]))
            if frames <= 0:
                return torch.cat([current, next_batch], dim=0), mode, 0
            a = current[-frames:]
            b = next_batch[:frames]
            progress = self._progress(frames, current.device, current.dtype, easing)
            transition_frames = self._render_transition(a, b, transition, progress, wipe_softness, effect_intensity)
            return torch.cat([current[:-frames], transition_frames, next_batch[frames:]], dim=0).clamp(0.0, 1.0), mode, frames

        frames = int(requested_frames)
        if frames <= 0:
            return torch.cat([current, next_batch], dim=0), mode, 0
        a = current[-1:].expand(frames, -1, -1, -1)
        b = next_batch[:1].expand(frames, -1, -1, -1)
        progress = self._progress(frames, current.device, current.dtype, easing)
        transition_frames = self._render_transition(a, b, transition, progress, wipe_softness, effect_intensity)
        return torch.cat([current, transition_frames, next_batch], dim=0).clamp(0.0, 1.0), mode, frames

    def _merge_audio_pair(
        self,
        current_audio,
        next_audio,
        current_frame_count,
        next_frame_count,
        resolved_mode,
        actual_transition_frames,
        transition,
        easing,
        fps,
        sample_rate,
    ):
        if transition == "cut" or actual_transition_frames <= 0:
            merged = torch.cat([current_audio, next_audio], dim=-1)
            return self._fit_audio_to_frames(merged, current_frame_count + next_frame_count, fps, sample_rate)

        if resolved_mode == "overlap":
            pre_audio = self._audio_slice_for_frames(
                current_audio,
                0,
                current_frame_count - actual_transition_frames,
                fps,
                sample_rate,
            )
            outgoing_audio = self._audio_slice_for_frames(
                current_audio,
                current_frame_count - actual_transition_frames,
                current_frame_count,
                fps,
                sample_rate,
            )
            incoming_audio = self._audio_slice_for_frames(
                next_audio,
                0,
                actual_transition_frames,
                fps,
                sample_rate,
            )
            post_audio = self._audio_slice_for_frames(
                next_audio,
                actual_transition_frames,
                next_frame_count,
                fps,
                sample_rate,
            )
            overlap_audio = self._mix_overlap_audio(outgoing_audio, incoming_audio, easing)
            merged = torch.cat([pre_audio, overlap_audio, post_audio], dim=-1)
            return self._fit_audio_to_frames(
                merged,
                current_frame_count + next_frame_count - actual_transition_frames,
                fps,
                sample_rate,
            )

        transition_silence = self._silence_for_frames(actual_transition_frames, current_audio, fps, sample_rate)
        merged = torch.cat([current_audio, transition_silence, next_audio], dim=-1)
        return self._fit_audio_to_frames(
            merged,
            current_frame_count + actual_transition_frames + next_frame_count,
            fps,
            sample_rate,
        )

    def merge_transitions(
        self,
        image_batch_1,
        image_batch_2,
        inputcount,
        fps,
        transition_mode,
        default_transition,
        default_transition_frames,
        easing,
        size_match,
        resize_method,
        wipe_softness,
        effect_intensity,
        audio_1=None,
        audio_2=None,
        transition_plan="",
        **kwargs,
    ):
        if "batch_count" in kwargs:
            inputcount = kwargs["batch_count"]
        inputcount = max(2, min(int(inputcount), self.MAX_BATCHES))
        fps = float(fps)
        if fps <= 0:
            raise ValueError("fps must be greater than 0.")

        batches = [
            self._validate_batch(image_batch_1, "image_batch_1"),
            self._validate_batch(image_batch_2, "image_batch_2"),
        ]
        audios = [
            self._coerce_audio_input(audio_1, "audio_1"),
            self._coerce_audio_input(audio_2, "audio_2"),
        ]
        for index in range(3, inputcount + 1):
            batch = kwargs.get(f"image_batch_{index}")
            if batch is None:
                raise ValueError(
                    f"image_batch_{index} must be connected when inputcount is {inputcount}. "
                    "Set inputcount, click Update inputs, then connect the added slot."
                )
            batches.append(self._validate_batch(batch, f"image_batch_{index}"))
            audios.append(self._coerce_audio_input(kwargs.get(f"audio_{index}"), f"audio_{index}"))

        batches = self._prepare_batches(batches, size_match, resize_method)
        transition_overrides = self._parse_transition_plan(transition_plan)
        input_frame_counts = [int(batch.shape[0]) for batch in batches]
        sample_rate = self._choose_audio_sample_rate(audios)
        audio_clips = self._prepare_audio_clips(audios, input_frame_counts, fps, sample_rate)

        output = batches[0]
        output_audio = audio_clips[0]
        info = [
            f"batches={inputcount}",
            f"input_frames={input_frame_counts}",
            f"size={int(output.shape[2])}x{int(output.shape[1])}",
            f"fps={fps:g}",
            f"audio_sample_rate={sample_rate}",
        ]

        for pair_index, next_batch in enumerate(batches[1:], start=1):
            current_frame_count = int(output.shape[0])
            next_frame_count = int(next_batch.shape[0])

            if pair_index in transition_overrides:
                transition, plan_frames = transition_overrides[pair_index]
                pair_frames = plan_frames if plan_frames is not None else int(kwargs.get(f"transition_frames_{pair_index}", 0) or 0)
            else:
                transition = kwargs.get(f"transition_{pair_index}", "same_as_default")
                pair_frames = int(kwargs.get(f"transition_frames_{pair_index}", 0) or 0)

            transition = self._normalize_transition_name(transition)
            if transition == "same_as_default":
                transition = self._normalize_transition_name(default_transition)
            if transition not in self.TRANSITIONS:
                raise ValueError(
                    f"Invalid transition '{transition}' for pair {pair_index}. "
                    f"Valid transitions: {', '.join(self.TRANSITIONS)}"
                )

            requested_frames = pair_frames if pair_frames > 0 else int(default_transition_frames)
            output, resolved_mode, actual_frames = self._merge_pair(
                output,
                next_batch,
                transition,
                requested_frames,
                transition_mode,
                easing,
                wipe_softness,
                effect_intensity,
            )
            output_audio = self._merge_audio_pair(
                output_audio,
                audio_clips[pair_index],
                current_frame_count,
                next_frame_count,
                resolved_mode,
                actual_frames,
                transition,
                easing,
                fps,
                sample_rate,
            )
            info.append(
                f"{pair_index}->{pair_index + 1}: transition={transition}, mode={resolved_mode}, "
                f"frames={actual_frames}, output_frames={int(output.shape[0])}"
            )

        frame_count = int(output.shape[0])
        output_audio = self._fit_audio_to_frames(output_audio, frame_count, fps, sample_rate)
        info.append(f"final_frames={frame_count}")
        info.append(f"audio_samples={int(output_audio.shape[-1])}")
        return (
            output.contiguous(),
            {"waveform": output_audio.contiguous(), "sample_rate": sample_rate},
            fps,
            frame_count,
            "\n".join(info),
        )


class VideoFilterBatch:
    """Applies video-style color presets and effects to a ComfyUI image batch."""

    FILTER_PRESETS = [
        "none",
        "sunset_glow",
        "moonlight",
        "cyberpunk",
        "film_noir",
        "dreamy",
        "retro_vhs",
        "teal_orange",
        "bleach_bypass",
        "cross_process",
        "faded_film",
        "warm_cinematic",
        "cool_cinematic",
        "vintage_chrome",
        "golden_hour",
        "sepia",
        "kodak_portra",
        "fuji_velvia",
        "polaroid",
        "horror_green",
        "arctic_blue",
        "desert_heat",
        "matrix_green",
    ]

    EFFECTS = [
        "none",
        "sharpen",
        "chromatic_aberration",
        "vignette",
        "film_grain",
        "glitch",
        "rgb_split",
        "lens_distortion",
        "motion_blur",
        "posterize",
        "duotone",
        "bloom",
        "scanlines",
    ]

    PRESET_CONFIG = {
        "sunset_glow": {
            "exposure": 0.08,
            "contrast": 1.08,
            "saturation": 1.22,
            "temperature": 0.52,
            "tint": 0.08,
            "highlights": ((1.0, 0.52, 0.18), 0.12),
            "shadows": ((0.30, 0.08, 0.18), 0.04),
        },
        "moonlight": {
            "exposure": -0.05,
            "contrast": 1.06,
            "saturation": 0.74,
            "temperature": -0.58,
            "tint": -0.08,
            "shadows": ((0.10, 0.20, 0.62), 0.16),
            "highlights": ((0.58, 0.72, 1.0), 0.06),
        },
        "cyberpunk": {
            "exposure": 0.03,
            "contrast": 1.24,
            "saturation": 1.45,
            "temperature": -0.18,
            "tint": 0.38,
            "shadows": ((0.00, 0.72, 1.0), 0.12),
            "highlights": ((1.0, 0.04, 0.72), 0.14),
            "curve": 0.20,
        },
        "film_noir": {
            "exposure": -0.03,
            "contrast": 1.36,
            "saturation": 0.0,
            "fade": 0.04,
            "curve": 0.30,
        },
        "dreamy": {
            "exposure": 0.12,
            "contrast": 0.86,
            "saturation": 1.08,
            "temperature": 0.18,
            "fade": 0.18,
            "soft_glow": 0.28,
        },
        "retro_vhs": {
            "exposure": -0.02,
            "contrast": 1.10,
            "saturation": 0.82,
            "temperature": 0.16,
            "tint": 0.16,
            "fade": 0.12,
            "channel_multipliers": (1.08, 0.96, 1.03),
            "scanline_mix": 0.06,
        },
        "teal_orange": {
            "contrast": 1.12,
            "saturation": 1.16,
            "shadows": ((0.00, 0.52, 0.58), 0.14),
            "midtones": ((0.96, 0.46, 0.12), 0.04),
            "highlights": ((1.0, 0.56, 0.18), 0.10),
        },
        "bleach_bypass": {
            "exposure": -0.02,
            "contrast": 1.32,
            "saturation": 0.48,
            "curve": 0.20,
        },
        "cross_process": {
            "contrast": 1.18,
            "saturation": 1.28,
            "temperature": 0.12,
            "tint": -0.20,
            "channel_multipliers": (1.10, 1.04, 0.92),
            "shadows": ((0.04, 0.20, 0.60), 0.08),
            "highlights": ((0.98, 0.88, 0.28), 0.10),
        },
        "faded_film": {
            "exposure": 0.04,
            "contrast": 0.88,
            "saturation": 0.82,
            "temperature": 0.14,
            "fade": 0.28,
            "shadows": ((0.14, 0.12, 0.10), 0.04),
        },
        "warm_cinematic": {
            "exposure": 0.03,
            "contrast": 1.16,
            "saturation": 1.06,
            "temperature": 0.30,
            "shadows": ((0.08, 0.20, 0.32), 0.07),
            "highlights": ((1.0, 0.72, 0.38), 0.08),
        },
        "cool_cinematic": {
            "exposure": -0.02,
            "contrast": 1.16,
            "saturation": 0.98,
            "temperature": -0.34,
            "shadows": ((0.04, 0.18, 0.42), 0.12),
            "highlights": ((0.76, 0.90, 1.0), 0.04),
        },
        "vintage_chrome": {
            "contrast": 1.18,
            "saturation": 1.32,
            "temperature": 0.20,
            "tint": -0.04,
            "fade": 0.08,
            "curve": 0.18,
        },
        "golden_hour": {
            "exposure": 0.08,
            "contrast": 1.08,
            "saturation": 1.18,
            "temperature": 0.44,
            "highlights": ((1.0, 0.68, 0.22), 0.14),
            "shadows": ((0.22, 0.10, 0.04), 0.04),
        },
        "sepia": {
            "contrast": 1.04,
            "saturation": 0.42,
            "temperature": 0.42,
            "channel_multipliers": (1.16, 1.00, 0.78),
            "fade": 0.10,
        },
        "kodak_portra": {
            "exposure": 0.05,
            "contrast": 0.96,
            "saturation": 1.10,
            "temperature": 0.16,
            "tint": 0.05,
            "highlights": ((1.0, 0.78, 0.58), 0.05),
        },
        "fuji_velvia": {
            "contrast": 1.20,
            "saturation": 1.42,
            "temperature": -0.04,
            "curve": 0.22,
            "shadows": ((0.04, 0.18, 0.26), 0.05),
        },
        "polaroid": {
            "exposure": 0.10,
            "contrast": 0.90,
            "saturation": 0.92,
            "temperature": 0.22,
            "fade": 0.22,
            "highlights": ((1.0, 0.88, 0.62), 0.06),
        },
        "horror_green": {
            "exposure": -0.08,
            "contrast": 1.26,
            "saturation": 0.72,
            "temperature": -0.26,
            "tint": -0.48,
            "shadows": ((0.00, 0.24, 0.08), 0.18),
            "highlights": ((0.68, 1.0, 0.50), 0.05),
        },
        "arctic_blue": {
            "exposure": 0.03,
            "contrast": 1.06,
            "saturation": 0.84,
            "temperature": -0.62,
            "shadows": ((0.06, 0.22, 0.56), 0.14),
            "highlights": ((0.72, 0.92, 1.0), 0.08),
        },
        "desert_heat": {
            "exposure": 0.08,
            "contrast": 1.14,
            "saturation": 1.12,
            "temperature": 0.58,
            "tint": -0.06,
            "shadows": ((0.36, 0.12, 0.02), 0.07),
            "highlights": ((1.0, 0.72, 0.28), 0.12),
        },
        "matrix_green": {
            "exposure": -0.04,
            "contrast": 1.18,
            "saturation": 0.66,
            "temperature": -0.36,
            "tint": -0.52,
            "channel_multipliers": (0.74, 1.22, 0.78),
            "shadows": ((0.00, 0.20, 0.04), 0.16),
        },
    }

    EFFECT_ALIASES = {
        "ca": "chromatic_aberration",
        "chromatic": "chromatic_aberration",
        "grain": "film_grain",
        "vhs_glitch": "glitch",
        "rgb": "rgb_split",
        "rgb_shift": "rgb_split",
        "distortion": "lens_distortion",
        "blur": "motion_blur",
        "motion": "motion_blur",
        "posterize_color": "posterize",
        "duo_tone": "duotone",
        "scanline": "scanlines",
    }

    PRESET_ALIASES = {
        "teal_and_orange": "teal_orange",
        "tealorange": "teal_orange",
        "black_and_white": "film_noir",
        "bw": "film_noir",
        "noir": "film_noir",
        "vhs": "retro_vhs",
        "faded": "faded_film",
        "portra": "kodak_portra",
        "velvia": "fuji_velvia",
        "horror": "horror_green",
        "arctic": "arctic_blue",
        "desert": "desert_heat",
        "matrix": "matrix_green",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input ComfyUI IMAGE batch [frames, height, width, channels]."
                }),
                "filter_preset": (cls.FILTER_PRESETS, {
                    "default": "none",
                    "tooltip": "Color grade preset. Use none when only applying effects."
                }),
                "preset_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Preset strength. 1.0 is the designed look; values above 1 exaggerate it."
                }),
                "effect_1": (cls.EFFECTS, {
                    "default": "none",
                    "tooltip": "First optional effect in the chain."
                }),
                "effect_1_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Strength for effect_1."
                }),
                "effect_2": (cls.EFFECTS, {
                    "default": "none",
                    "tooltip": "Second optional effect in the chain."
                }),
                "effect_2_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Strength for effect_2."
                }),
                "effect_3": (cls.EFFECTS, {
                    "default": "none",
                    "tooltip": "Third optional effect in the chain."
                }),
                "effect_3_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Strength for effect_3."
                }),
                "global_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Final wet/dry mix between the original image batch and the filtered result."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for grain and glitch effects."
                }),
            },
            "optional": {
                "effect_stack": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Optional extra effect chain, one per line. Examples:\n"
                        "film_grain:0.35\n"
                        "vignette:0.8\n"
                        "chromatic-aberration:0.5"
                    )
                }),
                "duotone_shadow": ("STRING", {
                    "default": "#161026",
                    "tooltip": "Shadow color for the duotone effect as #RRGGBB."
                }),
                "duotone_highlight": ("STRING", {
                    "default": "#ffd166",
                    "tooltip": "Highlight color for the duotone effect as #RRGGBB."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "filter_info")
    FUNCTION = "apply_filters"
    CATEGORY = "Image/Animation"

    @staticmethod
    def _normalize_name(value, aliases):
        normalized = str(value or "none").strip().lower()
        normalized = re.sub(r"[\s\-]+", "_", normalized)
        return aliases.get(normalized, normalized)

    @staticmethod
    def _mix(a, b, alpha):
        return a * (1.0 - alpha) + b * alpha

    @staticmethod
    def _validate_images(images):
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor in ComfyUI IMAGE format.")
        if images.ndim != 4:
            raise ValueError("images must be a 4D tensor: [frames, height, width, channels].")
        if int(images.shape[-1]) < 3:
            raise ValueError("images must have at least 3 color channels.")
        if int(images.shape[0]) < 1:
            raise ValueError("images must contain at least one frame.")
        return images.float().clamp(0.0, 1.0)

    @staticmethod
    def _luma(rgb):
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, 3)
        return (rgb * weights).sum(dim=-1, keepdim=True)

    @staticmethod
    def _color_tensor(rgb, color):
        return torch.tensor(color, device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, 3)

    @staticmethod
    def _uv_fields(images):
        n, h, w, _ = images.shape
        xs = torch.linspace(0.0, 1.0, w, device=images.device, dtype=images.dtype).view(1, 1, w, 1)
        ys = torch.linspace(0.0, 1.0, h, device=images.device, dtype=images.dtype).view(1, h, 1, 1)
        return xs.expand(n, h, w, 1), ys.expand(n, h, w, 1)

    @staticmethod
    def _frame_index_field(images):
        n = images.shape[0]
        return torch.arange(n, device=images.device, dtype=images.dtype).view(n, 1, 1, 1)

    @staticmethod
    def _pseudo_noise(field):
        return torch.remainder(torch.sin(field) * 43758.5453123, 1.0)

    @staticmethod
    def _parse_hex_color(value, fallback):
        text = str(value or "").strip()
        if text.startswith("#"):
            text = text[1:]
        if len(text) == 3:
            text = "".join(ch * 2 for ch in text)
        if len(text) != 6:
            return fallback
        try:
            return tuple(int(text[index:index + 2], 16) / 255.0 for index in (0, 2, 4))
        except ValueError:
            return fallback

    def _sample_rgb(self, rgb, sample_x, sample_y, padding_mode="border"):
        n, _, _, _ = rgb.shape
        if sample_x.shape[0] == 1 and n > 1:
            sample_x = sample_x.expand(n, -1, -1, -1)
        if sample_y.shape[0] == 1 and n > 1:
            sample_y = sample_y.expand(n, -1, -1, -1)
        grid = torch.cat([sample_x * 2.0 - 1.0, sample_y * 2.0 - 1.0], dim=-1)
        sampled = torch_nn_func.grid_sample(
            rgb.permute(0, 3, 1, 2),
            grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=True,
        )
        return sampled.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _bounds_mask(sample_x, sample_y, n):
        mask = ((sample_x >= 0.0) & (sample_x <= 1.0) & (sample_y >= 0.0) & (sample_y <= 1.0)).to(sample_x.dtype)
        if mask.shape[0] == 1 and n > 1:
            mask = mask.expand(n, -1, -1, -1)
        return mask

    @staticmethod
    def _box_blur(rgb, radius):
        radius = int(max(0, min(radius, 48)))
        if radius == 0:
            return rgb
        kernel = radius * 2 + 1
        x = rgb.permute(0, 3, 1, 2)
        x = torch_nn_func.pad(x, (radius, radius, radius, radius), mode="replicate")
        x = torch_nn_func.avg_pool2d(x, kernel_size=kernel, stride=1)
        return x.permute(0, 2, 3, 1).contiguous()

    def _random_like(self, rgb, seed):
        try:
            generator = torch.Generator(device=rgb.device)
            generator.manual_seed(int(seed) & 0x7fffffffffffffff)
            return torch.rand(rgb.shape, device=rgb.device, dtype=rgb.dtype, generator=generator)
        except (TypeError, RuntimeError):
            torch.manual_seed(int(seed) & 0x7fffffff)
            return torch.rand_like(rgb)

    def _apply_tonal_grade(self, rgb, config, strength):
        strength = min(max(float(strength), 0.0), 2.0)
        if strength == 0.0:
            return rgb

        exposure = float(config.get("exposure", 0.0)) * strength
        contrast = 1.0 + (float(config.get("contrast", 1.0)) - 1.0) * strength
        saturation = 1.0 + (float(config.get("saturation", 1.0)) - 1.0) * strength
        temperature = float(config.get("temperature", 0.0)) * strength
        tint = float(config.get("tint", 0.0)) * strength
        fade = min(max(float(config.get("fade", 0.0)) * strength, 0.0), 0.95)
        curve = min(max(float(config.get("curve", 0.0)) * strength, -1.0), 1.0)

        out = rgb * float(2.0 ** exposure)

        temp_vector = torch.tensor(
            [
                1.0 + 0.22 * temperature + 0.08 * tint,
                1.0 - 0.16 * tint,
                1.0 - 0.22 * temperature + 0.08 * tint,
            ],
            device=rgb.device,
            dtype=rgb.dtype,
        ).view(1, 1, 1, 3)
        out = out * temp_vector

        if "channel_multipliers" in config:
            multipliers = torch.tensor(config["channel_multipliers"], device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, 3)
            multipliers = 1.0 + (multipliers - 1.0) * strength
            out = out * multipliers

        out = (out - 0.5) * contrast + 0.5
        gray = self._luma(out)
        out = self._mix(gray, out, saturation)

        if fade > 0.0:
            out = out * (1.0 - 0.34 * fade) + 0.10 * fade

        out = out.clamp(0.0, 1.0)
        if curve > 0.0:
            s_curve = out * out * (3.0 - 2.0 * out)
            out = self._mix(out, s_curve, curve)
        elif curve < 0.0:
            soft_curve = torch.sqrt(out.clamp(0.0, 1.0))
            out = self._mix(out, soft_curve, -curve)

        return self._apply_color_balance(out, config, strength).clamp(0.0, 1.0)

    def _apply_color_balance(self, rgb, config, strength):
        luma = self._luma(rgb).clamp(0.0, 1.0)
        masks = {
            "shadows": torch.pow(1.0 - luma, 2.0),
            "midtones": torch.pow((1.0 - torch.abs(luma - 0.5) * 2.0).clamp(0.0, 1.0), 2.0),
            "highlights": torch.pow(luma, 2.0),
        }
        out = rgb
        for key, mask in masks.items():
            if key not in config:
                continue
            color, amount = config[key]
            amount = float(amount) * min(max(float(strength), 0.0), 2.0)
            color_tensor = self._color_tensor(rgb, color)
            out = out + (color_tensor - 0.5) * 2.0 * mask * amount
        return out

    def _apply_preset(self, rgb, preset, strength):
        preset = self._normalize_name(preset, self.PRESET_ALIASES)
        if preset == "none" or float(strength) <= 0.0:
            return rgb
        if preset not in self.PRESET_CONFIG:
            raise ValueError(f"Invalid filter_preset '{preset}'. Valid presets: {', '.join(self.FILTER_PRESETS)}")

        config = self.PRESET_CONFIG[preset]
        out = self._apply_tonal_grade(rgb, config, strength)

        glow_mix = float(config.get("soft_glow", 0.0)) * min(max(float(strength), 0.0), 2.0)
        if glow_mix > 0.0:
            blur = self._box_blur(out, radius=4)
            out = self._mix(out, blur + 0.04, min(glow_mix, 0.75)).clamp(0.0, 1.0)

        scanline_mix = float(config.get("scanline_mix", 0.0)) * min(max(float(strength), 0.0), 2.0)
        if scanline_mix > 0.0:
            out = self._effect_scanlines(out, scanline_mix)

        return out.clamp(0.0, 1.0)

    def _effect_sharpen(self, rgb, strength):
        amount = min(max(float(strength), 0.0), 3.0)
        blur = self._box_blur(rgb, radius=max(1, int(round(1 + amount))))
        return (rgb + (rgb - blur) * (0.42 * amount)).clamp(0.0, 1.0)

    def _effect_chromatic_aberration(self, rgb, strength):
        amount = min(max(float(strength), 0.0), 3.0)
        xs, ys = self._uv_fields(rgb)
        dx = xs - 0.5
        dy = ys - 0.5
        dist = torch.sqrt(dx * dx + dy * dy).clamp(0.0, 1.0)
        offset = dist * (0.010 + 0.012 * amount) * amount
        red = self._sample_rgb(rgb, xs + dx * offset, ys + dy * offset)[..., 0:1]
        green = rgb[..., 1:2]
        blue = self._sample_rgb(rgb, xs - dx * offset, ys - dy * offset)[..., 2:3]
        return torch.cat([red, green, blue], dim=-1).clamp(0.0, 1.0)

    def _effect_vignette(self, rgb, strength):
        amount = min(max(float(strength), 0.0), 3.0)
        xs, ys = self._uv_fields(rgb)
        _, h, w, _ = rgb.shape
        aspect = float(w) / max(float(h), 1.0)
        dx = (xs - 0.5) * min(aspect, 1.0) * 2.0
        dy = (ys - 0.5) / max(aspect, 1.0) * 2.0
        radius = torch.sqrt(dx * dx + dy * dy)
        edge = ((radius - 0.38) / 0.72).clamp(0.0, 1.0)
        edge = edge * edge * (3.0 - 2.0 * edge)
        darken = min(0.82, 0.38 * amount)
        feather = (1.0 - edge * darken).clamp(0.0, 1.0)
        center_lift = (1.0 - edge) * min(0.06, 0.025 * amount)
        return (rgb * feather + center_lift).clamp(0.0, 1.0)

    def _effect_film_grain(self, rgb, strength, seed):
        amount = min(max(float(strength), 0.0), 3.0)
        noise = self._random_like(rgb, int(seed)) - 0.5
        luma = self._luma(rgb)
        shadow_bias = (1.25 - luma).clamp(0.25, 1.25)
        return (rgb + noise * shadow_bias * (0.075 * amount)).clamp(0.0, 1.0)

    def _effect_glitch(self, rgb, strength, seed):
        amount = min(max(float(strength), 0.0), 3.0)
        xs, ys = self._uv_fields(rgb)
        frame_index = self._frame_index_field(rgb)
        bands = torch.floor(ys * max(8, int(round(14 + amount * 10))))
        band_noise = self._pseudo_noise(bands * 12.9898 + frame_index * 78.233 + float(seed) * 0.017)
        active = (band_noise > (0.72 - 0.06 * amount)).to(rgb.dtype)
        offset = (band_noise - 0.5) * active * (0.045 * amount)
        shifted = self._sample_rgb(rgb, xs + offset, ys)

        split = 0.006 * amount
        red = self._sample_rgb(shifted, xs + split * active, ys)[..., 0:1]
        green = shifted[..., 1:2]
        blue = self._sample_rgb(shifted, xs - split * active, ys)[..., 2:3]
        out = torch.cat([red, green, blue], dim=-1)
        tear = active * self._pseudo_noise(bands * 41.7 + float(seed) * 0.13) * 0.10 * amount
        return (out + tear).clamp(0.0, 1.0)

    def _effect_rgb_split(self, rgb, strength):
        amount = min(max(float(strength), 0.0), 3.0)
        _, _, w, _ = rgb.shape
        xs, ys = self._uv_fields(rgb)
        shift = (1.0 + 2.5 * amount) / max(float(w), 1.0)
        red = self._sample_rgb(rgb, xs + shift, ys)[..., 0:1]
        green = rgb[..., 1:2]
        blue = self._sample_rgb(rgb, xs - shift, ys)[..., 2:3]
        return torch.cat([red, green, blue], dim=-1).clamp(0.0, 1.0)

    def _effect_lens_distortion(self, rgb, strength):
        amount = min(max(float(strength), 0.0), 3.0)
        n, _, _, _ = rgb.shape
        xs, ys = self._uv_fields(rgb)
        dx = xs - 0.5
        dy = ys - 0.5
        r2 = (dx * dx + dy * dy).clamp(0.0, 0.5)
        barrel = 0.30 * amount
        edge_pull = 1.0 + barrel * r2 + 0.16 * amount * r2 * r2
        zoom_compensation = 1.0 / (1.0 + 0.055 * amount)
        sample_x = 0.5 + dx * edge_pull * zoom_compensation
        sample_y = 0.5 + dy * edge_pull * zoom_compensation
        warped = self._sample_rgb(rgb, sample_x, sample_y, padding_mode="zeros")
        bounds = self._bounds_mask(sample_x, sample_y, n)
        edge_fade = (1.0 - torch.pow(r2 / 0.5, 1.6) * (0.14 * amount)).clamp(0.0, 1.0)
        return (warped * bounds * edge_fade).clamp(0.0, 1.0)

    def _effect_motion_blur(self, rgb, strength):
        amount = min(max(float(strength), 0.0), 3.0)
        frame_count = int(rgb.shape[0])
        if frame_count <= 1 or amount <= 0.0:
            return rgb

        indices = torch.arange(frame_count, device=rgb.device)
        radius = min(max(1, int(round(1 + amount * 2.0))), min(8, frame_count - 1))
        acc = rgb * (1.0 + radius)
        weight_sum = 1.0 + radius
        for offset in range(1, radius + 1):
            weight = float(radius + 1 - offset) / float(radius + 1)
            prev_index = (indices - offset).clamp(0, frame_count - 1)
            next_index = (indices + offset).clamp(0, frame_count - 1)
            acc = acc + rgb.index_select(0, prev_index) * weight
            acc = acc + rgb.index_select(0, next_index) * weight
            weight_sum += weight * 2.0

        temporal_blur = acc / float(weight_sum)
        prev_frame = rgb.index_select(0, (indices - 1).clamp(0, frame_count - 1))
        next_frame = rgb.index_select(0, (indices + 1).clamp(0, frame_count - 1))
        motion_delta = (next_frame - prev_frame).abs().mean(dim=-1, keepdim=True)
        motion_mask = (motion_delta * (5.5 + amount * 2.5)).clamp(0.0, 1.0)
        wet = min(0.92, 0.28 + amount * 0.20)
        return self._mix(rgb, temporal_blur, motion_mask * wet).clamp(0.0, 1.0)

    def _effect_posterize(self, rgb, strength):
        amount = min(max(float(strength), 0.0), 3.0)
        levels = max(2, int(round(12 - amount * 3.2)))
        poster = torch.round(rgb * float(levels - 1)) / float(levels - 1)
        return self._mix(rgb, poster, min(amount, 1.0)).clamp(0.0, 1.0)

    def _effect_duotone(self, rgb, strength, shadow_color, highlight_color):
        amount = min(max(float(strength), 0.0), 3.0)
        shadow = self._color_tensor(rgb, shadow_color)
        highlight = self._color_tensor(rgb, highlight_color)
        luma = self._luma(rgb).clamp(0.0, 1.0)
        mapped = shadow * (1.0 - luma) + highlight * luma
        return self._mix(rgb, mapped, min(amount, 1.0)).clamp(0.0, 1.0)

    def _effect_bloom(self, rgb, strength):
        amount = min(max(float(strength), 0.0), 3.0)
        threshold = 0.64
        bright = ((rgb - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
        blur = self._box_blur(bright, radius=max(3, int(round(4 + amount * 4))))
        return (rgb + blur * (0.18 * amount)).clamp(0.0, 1.0)

    def _effect_scanlines(self, rgb, strength):
        amount = min(max(float(strength), 0.0), 3.0)
        n, h, _, _ = rgb.shape
        rows = torch.arange(h, device=rgb.device, dtype=rgb.dtype).view(1, h, 1, 1)
        frame_index = self._frame_index_field(rgb)
        phase = torch.remainder(rows + frame_index, 2.0)
        dark_line = (phase < 1.0).to(rgb.dtype)
        fine_beam = 0.92 + 0.08 * torch.cos(rows * np.pi)
        line_mask = (1.0 - dark_line * min(0.62, 0.24 * amount)) * fine_beam
        rolling_bar = 1.0 - 0.035 * amount * torch.sin((rows / max(float(h), 1.0) + frame_index * 0.023) * np.pi * 2.0)
        return (rgb * line_mask.clamp(0.0, 1.0) * rolling_bar.clamp(0.82, 1.08)).clamp(0.0, 1.0)

    def _apply_effect(self, rgb, effect, strength, seed, duotone_shadow, duotone_highlight):
        effect = self._normalize_name(effect, self.EFFECT_ALIASES)
        if effect == "none" or float(strength) <= 0.0:
            return rgb
        if effect not in self.EFFECTS:
            raise ValueError(f"Invalid effect '{effect}'. Valid effects: {', '.join(self.EFFECTS)}")

        if effect == "sharpen":
            return self._effect_sharpen(rgb, strength)
        if effect == "chromatic_aberration":
            return self._effect_chromatic_aberration(rgb, strength)
        if effect == "vignette":
            return self._effect_vignette(rgb, strength)
        if effect == "film_grain":
            return self._effect_film_grain(rgb, strength, seed)
        if effect == "glitch":
            return self._effect_glitch(rgb, strength, seed)
        if effect == "rgb_split":
            return self._effect_rgb_split(rgb, strength)
        if effect == "lens_distortion":
            return self._effect_lens_distortion(rgb, strength)
        if effect == "motion_blur":
            return self._effect_motion_blur(rgb, strength)
        if effect == "posterize":
            return self._effect_posterize(rgb, strength)
        if effect == "duotone":
            return self._effect_duotone(rgb, strength, duotone_shadow, duotone_highlight)
        if effect == "bloom":
            return self._effect_bloom(rgb, strength)
        if effect == "scanlines":
            return self._effect_scanlines(rgb, strength)
        return rgb

    def _parse_effect_stack(self, effect_stack):
        entries = []
        for raw_entry in re.split(r"[\n;]+", str(effect_stack or "")):
            entry = raw_entry.strip()
            if not entry or entry.startswith("#"):
                continue
            parts = re.split(r"[:=,]", entry, maxsplit=1)
            effect = self._normalize_name(parts[0], self.EFFECT_ALIASES)
            if effect not in self.EFFECTS:
                raise ValueError(f"Invalid effect_stack entry '{entry}'. Valid effects: {', '.join(self.EFFECTS)}")
            strength = 1.0
            if len(parts) > 1:
                strength_text = parts[1].strip().split()[0] if parts[1].strip() else "1.0"
                try:
                    strength = float(strength_text)
                except ValueError as exc:
                    raise ValueError(f"Invalid strength in effect_stack entry '{entry}'.") from exc
            entries.append((effect, strength))
        return entries

    def apply_filters(
        self,
        images,
        filter_preset,
        preset_strength,
        effect_1,
        effect_1_strength,
        effect_2,
        effect_2_strength,
        effect_3,
        effect_3_strength,
        global_strength,
        seed,
        effect_stack="",
        duotone_shadow="#161026",
        duotone_highlight="#ffd166",
    ):
        source = self._validate_images(images)
        original_rgb = source[..., :3]
        tail = source[..., 3:] if int(source.shape[-1]) > 3 else None
        output_rgb = self._apply_preset(original_rgb, filter_preset, preset_strength)

        shadow_color = self._parse_hex_color(duotone_shadow, (0.086, 0.063, 0.149))
        highlight_color = self._parse_hex_color(duotone_highlight, (1.0, 0.820, 0.400))
        effect_entries = [
            (effect_1, effect_1_strength),
            (effect_2, effect_2_strength),
            (effect_3, effect_3_strength),
        ]
        effect_entries.extend(self._parse_effect_stack(effect_stack))

        applied_effects = []
        for index, (effect, strength) in enumerate(effect_entries):
            normalized_effect = self._normalize_name(effect, self.EFFECT_ALIASES)
            if normalized_effect == "none" or float(strength) <= 0.0:
                continue
            output_rgb = self._apply_effect(
                output_rgb,
                normalized_effect,
                strength,
                int(seed) + index * 1009,
                shadow_color,
                highlight_color,
            )
            applied_effects.append(f"{normalized_effect}:{float(strength):g}")

        output_rgb = self._mix(original_rgb, output_rgb, min(max(float(global_strength), 0.0), 1.0)).clamp(0.0, 1.0)
        if tail is not None:
            output = torch.cat([output_rgb, tail], dim=-1)
        else:
            output = output_rgb

        normalized_preset = self._normalize_name(filter_preset, self.PRESET_ALIASES)
        info = [
            f"preset={normalized_preset}",
            f"preset_strength={float(preset_strength):g}",
            f"effects={', '.join(applied_effects) if applied_effects else 'none'}",
            f"global_strength={float(global_strength):g}",
            f"frames={int(output.shape[0])}",
            f"size={int(output.shape[2])}x{int(output.shape[1])}",
        ]
        return (output.contiguous(), "\n".join(info))


class VideoTrimmer:
    """
    Trim and modulate the framerate of video / image batches with precise time/frame slicing
    and sample-accurate audio synchronization (no audio skew).
    """

    TRIM_MODES = [
        "all",
        "start_duration",
        "start_end",
        "first_n",
        "last_n",
        "center_n",
        "cut_inside",
    ]

    TRIM_UNITS = [
        "seconds",
        "source_frames",
        "output_frames",
        "percentage",
    ]

    FRAMERATE_MODES = [
        "keep_source",
        "custom_fps",
        "multiplier",
        "target_frame_count",
    ]

    INTERPOLATIONS = [
        "nearest",
        "linear",
        "cubic",
    ]

    LOOP_MODES = [
        "clamp",
        "repeat",
        "mirror",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trim_mode": (cls.TRIM_MODES, {
                    "default": "last_n",
                    "tooltip": (
                        "Trimming mode:\n"
                        "• all: Keep entire video (useful for pure framerate conversion/resampling)\n"
                        "• start_duration: Trim starting at 'start_value' for duration 'duration_or_count'\n"
                        "• start_end: Trim between 'start_value' and 'end_value'\n"
                        "• first_n: Extract first N units (frames/seconds)\n"
                        "• last_n: Extract last N units (frames/seconds)\n"
                        "• center_n: Extract center N units around the midpoint\n"
                        "• cut_inside: Remove segment between start_value and end_value, splicing remaining parts"
                    )
                }),
                "trim_units": (cls.TRIM_UNITS, {
                    "default": "source_frames",
                    "tooltip": (
                        "Units for trim parameters:\n"
                        "• seconds: Real-world time in seconds\n"
                        "• source_frames: Frame count / indices based on source framerate\n"
                        "• output_frames: Frame count based on target output framerate\n"
                        "• percentage: 0% to 100% (or 0.0 to 1.0) of the video duration"
                    )
                }),
                "start_value": ("FLOAT", {
                    "default": 0.0,
                    "min": -100000.0,
                    "max": 100000.0,
                    "step": 0.01,
                    "tooltip": "Start position (can be negative to index from the end, e.g., -2.0s or -48 frames)."
                }),
                "duration_or_count": ("FLOAT", {
                    "default": 24.0,
                    "min": 0.0,
                    "max": 100000.0,
                    "step": 0.01,
                    "tooltip": "Length/count to extract (0 = until the end of video in start_duration mode)."
                }),
                "end_value": ("FLOAT", {
                    "default": 0.0,
                    "min": -100000.0,
                    "max": 100000.0,
                    "step": 0.01,
                    "tooltip": "End position for start_end / cut_inside modes (0 = end of video; negative = offset from end)."
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 0.01,
                    "max": 1000.0,
                    "step": 0.01,
                    "tooltip": "Source video framerate. Auto-detected when native 'video' input is connected unless overridden."
                }),
                "framerate_mode": (cls.FRAMERATE_MODES, {
                    "default": "keep_source",
                    "tooltip": (
                        "Framerate / resampling mode:\n"
                        "• keep_source: Preserve source framerate\n"
                        "• custom_fps: Resample frames to target_fps while preserving real-world playback speed and audio sync\n"
                        "• multiplier: Multiply source framerate by fps_multiplier\n"
                        "• target_frame_count: Produce exact target_frame_count output frames (auto-calculates output fps)"
                    )
                }),
                "target_fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 0.01,
                    "max": 1000.0,
                    "step": 0.01,
                    "tooltip": "Target framerate used when framerate_mode is 'custom_fps'."
                }),
                "fps_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Framerate scaling factor used when framerate_mode is 'multiplier' (e.g., 0.5 = half fps, 2.0 = double fps)."
                }),
                "target_frame_count": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Exact number of output frames when framerate_mode is 'target_frame_count'."
                }),
                "interpolation": (cls.INTERPOLATIONS, {
                    "default": "nearest",
                    "tooltip": (
                        "Frame interpolation method:\n"
                        "• nearest: Frame skipping/duplication (crisp, zero motion blur/ghosting)\n"
                        "• linear: Linear blend between adjacent frames\n"
                        "• cubic: Smooth cubic interpolation"
                    )
                }),
                "loop_mode": (cls.LOOP_MODES, {
                    "default": "clamp",
                    "tooltip": "How to handle boundary frames (clamp to edge, repeat, or mirror)."
                }),
            },
            "optional": {
                "video": ("VIDEO", {
                    "tooltip": "Optional native ComfyUI Video object (e.g. from LoadVideo or CreateVideo)."
                }),
                "images": ("IMAGE", {
                    "tooltip": "Optional batch of images [B, H, W, C]."
                }),
                "audio": ("AUDIO,VHS_AUDIO", {
                    "tooltip": "Optional audio track (ComfyUI AUDIO dict or VHS_AUDIO)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count", "trim_info")
    FUNCTION = "trim_video"
    CATEGORY = "Image/Animation"

    def _coerce_audio(self, audio, name="audio"):
        if audio is None:
            return None
        if isinstance(audio, Mapping):
            if "waveform" in audio and "sample_rate" in audio:
                return {"waveform": audio["waveform"], "sample_rate": audio["sample_rate"]}
            return audio
        if callable(audio):
            try:
                payload = audio()
            except Exception as e:
                print(f"[VideoTrimmer] Failed to call VHS_AUDIO callable: {e}")
                return None
            if not payload:
                return None
            try:
                import av
                if isinstance(payload, bytes):
                    container_input = io.BytesIO(payload)
                elif isinstance(payload, str) and os.path.exists(payload):
                    container_input = payload
                else:
                    container_input = io.BytesIO(payload)
                with av.open(container_input, mode="r") as container:
                    if not container.streams.audio:
                        return None
                    stream = container.streams.audio[0]
                    sample_rate = int(stream.codec_context.sample_rate) if stream.codec_context.sample_rate else 44100
                    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sample_rate)
                    chunks = []
                    for frame in container.decode(streams=stream.index):
                        for r_frame in resampler.resample(frame):
                            chunks.append(torch.from_numpy(r_frame.to_ndarray().copy()))
                    for r_frame in resampler.resample(None):
                        chunks.append(torch.from_numpy(r_frame.to_ndarray().copy()))
                    if not chunks:
                        return None
                    waveform = torch.cat(chunks, dim=-1).unsqueeze(0).float()
                    return {"waveform": waveform, "sample_rate": sample_rate}
            except Exception as e:
                print(f"[VideoTrimmer] Failed to decode audio payload: {e}")
                return None
        return None

    def _extract_inputs(self, video=None, images=None, audio=None, source_fps=24.0):
        detected_fps = None
        extracted_images = None
        extracted_audio = None

        if video is not None:
            if hasattr(video, "get_components") and callable(video.get_components):
                try:
                    comps = video.get_components()
                    extracted_images = comps.images
                    extracted_audio = comps.audio
                    if comps.frame_rate:
                        detected_fps = float(comps.frame_rate)
                except Exception as e:
                    print(f"[VideoTrimmer] Warning: Failed to extract components from video: {e}")
            elif isinstance(video, Mapping):
                extracted_images = video.get("images")
                extracted_audio = video.get("audio")
                if "frame_rate" in video:
                    detected_fps = float(video["frame_rate"])
                elif "fps" in video:
                    detected_fps = float(video["fps"])

        if images is not None:
            extracted_images = images

        if audio is not None:
            extracted_audio = audio

        if extracted_images is None:
            raise ValueError("VideoTrimmer: No image or video input provided. Please connect 'video' or 'images'.")

        if not isinstance(extracted_images, torch.Tensor) or extracted_images.ndim != 4:
            raise ValueError(f"VideoTrimmer: 'images' must be a 4D torch.Tensor [frames, height, width, channels], got {type(extracted_images)}.")

        if extracted_images.shape[0] == 0:
            raise ValueError("VideoTrimmer: Input image batch contains 0 frames.")

        if detected_fps is not None and detected_fps > 0:
            if source_fps == 24.0:
                effective_source_fps = detected_fps
            else:
                effective_source_fps = float(source_fps)
        else:
            effective_source_fps = max(0.001, float(source_fps))

        return extracted_images, extracted_audio, effective_source_fps, detected_fps

    def _to_seconds(self, val, units, input_fps, target_fps, total_duration):
        if units == "seconds":
            return float(val)
        elif units == "source_frames":
            return float(val) / input_fps
        elif units == "output_frames":
            return float(val) / target_fps
        elif units == "percentage":
            p = float(val)
            if abs(p) > 1.0 and abs(p) <= 100.0:
                p = p / 100.0
            return p * total_duration
        return float(val)

    def _get_frame_at_position(self, images, position, loop_mode="clamp"):
        total_frames = images.shape[0]
        if total_frames == 1:
            return images[0]
        if loop_mode == "clamp":
            idx = max(0, min(total_frames - 1, int(position)))
            return images[idx]
        elif loop_mode == "repeat":
            idx = int(position) % total_frames
            return images[idx]
        elif loop_mode == "mirror":
            cycle_length = (total_frames - 1) * 2
            if cycle_length <= 0:
                return images[0]
            pos_in_cycle = int(position) % cycle_length
            if pos_in_cycle < total_frames:
                return images[pos_in_cycle]
            else:
                mirror_idx = cycle_length - pos_in_cycle
                return images[mirror_idx]
        return images[0]

    def _interpolate_positions(self, images, positions, interpolation="nearest", loop_mode="clamp"):
        total_frames = images.shape[0]
        if total_frames == 0 or len(positions) == 0:
            return images[:0]

        device = images.device
        dtype = images.dtype
        num_out = len(positions)
        _, h, w, c = images.shape

        if interpolation == "nearest":
            indices = []
            for pos in positions:
                idx = int(round(pos))
                if loop_mode == "clamp":
                    idx = max(0, min(total_frames - 1, idx))
                elif loop_mode == "repeat":
                    idx = idx % total_frames
                elif loop_mode == "mirror":
                    cycle = (total_frames - 1) * 2
                    if cycle > 0:
                        p = idx % cycle
                        idx = p if p < total_frames else cycle - p
                    else:
                        idx = 0
                indices.append(idx)
            return images[indices]

        result = torch.zeros((num_out, h, w, c), dtype=dtype, device=device)
        for i, pos in enumerate(positions):
            if interpolation == "linear":
                pos_floor = int(np.floor(pos))
                pos_ceil = pos_floor + 1
                alpha = float(pos - pos_floor)
                f_a = self._get_frame_at_position(images, pos_floor, loop_mode)
                f_b = self._get_frame_at_position(images, pos_ceil, loop_mode)
                result[i] = f_a * (1.0 - alpha) + f_b * alpha
            elif interpolation == "cubic":
                pos_int = int(np.floor(pos))
                alpha = float(pos - pos_int)
                p0 = self._get_frame_at_position(images, pos_int - 1, loop_mode)
                p1 = self._get_frame_at_position(images, pos_int, loop_mode)
                p2 = self._get_frame_at_position(images, pos_int + 1, loop_mode)
                p3 = self._get_frame_at_position(images, pos_int + 2, loop_mode)
                a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
                b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3
                c = -0.5 * p0 + 0.5 * p2
                d = p1
                result[i] = torch.clamp(a * (alpha ** 3) + b * (alpha ** 2) + c * alpha + d, 0.0, 1.0)
            else:
                result[i] = self._get_frame_at_position(images, int(round(pos)), loop_mode)
        return result

    def _slice_audio(self, audio, intervals):
        if audio is None:
            return None

        coerced = self._coerce_audio(audio)
        if coerced is None:
            return None

        waveform = coerced["waveform"]
        if not isinstance(waveform, torch.Tensor):
            return None

        sr = int(coerced.get("sample_rate", 44100))
        if sr <= 0:
            sr = 44100

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)

        total_samples = waveform.shape[-1]
        sliced_pieces = []

        for t_start, t_end in intervals:
            t_dur = max(0.0, t_end - t_start)
            expected_samples = max(0, int(round(t_dur * sr)))
            if expected_samples == 0:
                continue

            start_samp = max(0, int(round(t_start * sr)))
            end_samp = min(total_samples, int(round(t_end * sr)))

            if start_samp >= total_samples:
                piece = torch.zeros((1, waveform.shape[1], expected_samples), dtype=waveform.dtype, device=waveform.device)
            else:
                actual = waveform[..., start_samp:end_samp]
                if actual.shape[-1] < expected_samples:
                    pad_len = expected_samples - actual.shape[-1]
                    pad = torch.zeros((1, waveform.shape[1], pad_len), dtype=waveform.dtype, device=waveform.device)
                    piece = torch.cat([actual, pad], dim=-1)
                elif actual.shape[-1] > expected_samples:
                    piece = actual[..., :expected_samples]
                else:
                    piece = actual
            sliced_pieces.append(piece)

        if not sliced_pieces:
            return None

        if len(sliced_pieces) == 1:
            final_waveform = sliced_pieces[0]
        else:
            final_waveform = torch.cat(sliced_pieces, dim=-1)

        return {"waveform": final_waveform, "sample_rate": sr}

    def trim_video(
        self,
        trim_mode,
        trim_units,
        start_value,
        duration_or_count,
        end_value,
        source_fps,
        framerate_mode,
        target_fps,
        fps_multiplier,
        target_frame_count,
        interpolation="nearest",
        loop_mode="clamp",
        video=None,
        images=None,
        audio=None,
    ):
        extracted_images, extracted_audio, input_fps, detected_fps = self._extract_inputs(
            video=video, images=images, audio=audio, source_fps=source_fps
        )

        total_frames = extracted_images.shape[0]
        total_duration = total_frames / input_fps

        if framerate_mode == "keep_source":
            out_fps = input_fps
        elif framerate_mode == "custom_fps":
            out_fps = max(0.01, float(target_fps))
        elif framerate_mode == "multiplier":
            out_fps = max(0.01, input_fps * max(0.001, float(fps_multiplier)))
        elif framerate_mode == "target_frame_count":
            out_fps = None

        ref_target_fps = out_fps if out_fps is not None else input_fps

        intervals = []

        if trim_mode == "all":
            intervals.append((0.0, total_duration))

        elif trim_mode == "start_duration":
            t_start = self._to_seconds(start_value, trim_units, input_fps, ref_target_fps, total_duration)
            if t_start < 0:
                t_start = max(0.0, total_duration + t_start)
            else:
                t_start = min(total_duration, max(0.0, t_start))

            if duration_or_count <= 0:
                t_end = total_duration
            else:
                t_dur = self._to_seconds(duration_or_count, trim_units, input_fps, ref_target_fps, total_duration)
                t_end = min(total_duration, t_start + max(0.0, t_dur))
            intervals.append((t_start, t_end))

        elif trim_mode == "start_end":
            t_start = self._to_seconds(start_value, trim_units, input_fps, ref_target_fps, total_duration)
            if t_start < 0:
                t_start = max(0.0, total_duration + t_start)
            else:
                t_start = min(total_duration, max(0.0, t_start))

            if end_value == 0.0 and start_value != 0.0:
                t_end = total_duration
            else:
                t_end = self._to_seconds(end_value, trim_units, input_fps, ref_target_fps, total_duration)
                if t_end < 0:
                    t_end = max(0.0, total_duration + t_end)
                elif t_end == 0.0:
                    t_end = total_duration
                else:
                    t_end = min(total_duration, max(0.0, t_end))

            if t_end < t_start:
                t_start, t_end = t_end, t_start
            intervals.append((t_start, t_end))

        elif trim_mode == "first_n":
            n_val = duration_or_count if duration_or_count > 0 else start_value
            t_len = max(0.0, self._to_seconds(n_val, trim_units, input_fps, ref_target_fps, total_duration))
            t_end = min(total_duration, t_len)
            intervals.append((0.0, t_end))

        elif trim_mode == "last_n":
            n_val = duration_or_count if duration_or_count > 0 else start_value
            t_len = max(0.0, self._to_seconds(n_val, trim_units, input_fps, ref_target_fps, total_duration))
            t_len = min(total_duration, t_len)
            t_start = max(0.0, total_duration - t_len)
            intervals.append((t_start, total_duration))

        elif trim_mode == "center_n":
            n_val = duration_or_count if duration_or_count > 0 else start_value
            t_len = max(0.0, self._to_seconds(n_val, trim_units, input_fps, ref_target_fps, total_duration))
            t_len = min(total_duration, t_len)
            t_start = max(0.0, (total_duration - t_len) / 2.0)
            t_end = min(total_duration, t_start + t_len)
            intervals.append((t_start, t_end))

        elif trim_mode == "cut_inside":
            t_cut_start = self._to_seconds(start_value, trim_units, input_fps, ref_target_fps, total_duration)
            if t_cut_start < 0:
                t_cut_start = max(0.0, total_duration + t_cut_start)
            else:
                t_cut_start = min(total_duration, max(0.0, t_cut_start))

            t_cut_end = self._to_seconds(end_value, trim_units, input_fps, ref_target_fps, total_duration)
            if t_cut_end < 0:
                t_cut_end = max(0.0, total_duration + t_cut_end)
            else:
                t_cut_end = min(total_duration, max(0.0, t_cut_end))

            if t_cut_end < t_cut_start:
                t_cut_start, t_cut_end = t_cut_end, t_cut_start

            if t_cut_start > 0.0:
                intervals.append((0.0, t_cut_start))
            if t_cut_end < total_duration:
                intervals.append((t_cut_end, total_duration))
            if len(intervals) == 0:
                intervals.append((0.0, min(total_duration, 1.0 / input_fps)))

        trimmed_duration = sum(max(0.0, t_e - t_s) for t_s, t_e in intervals)
        if trimmed_duration <= 0.0:
            trimmed_duration = 1.0 / input_fps

        if framerate_mode == "target_frame_count":
            target_count = max(1, int(target_frame_count))
            out_fps = max(0.01, float(target_count) / max(1e-6, trimmed_duration))
        else:
            out_fps = float(out_fps)

        output_frame_batches = []
        for t_s, t_e in intervals:
            t_seg = max(0.0, t_e - t_s)
            if t_seg <= 0.0:
                continue

            if framerate_mode == "target_frame_count" and len(intervals) == 1:
                n_frames = max(1, int(target_frame_count))
            else:
                n_frames = max(1, int(round(t_seg * out_fps)))

            positions = []
            for i in range(n_frames):
                t_pos = t_s + (i / out_fps)
                pos = t_pos * input_fps
                positions.append(pos)

            seg_frames = self._interpolate_positions(
                extracted_images, positions, interpolation=interpolation, loop_mode=loop_mode
            )
            output_frame_batches.append(seg_frames)

        if len(output_frame_batches) == 0:
            output_images = extracted_images[:1]
        elif len(output_frame_batches) == 1:
            output_images = output_frame_batches[0]
        else:
            output_images = torch.cat(output_frame_batches, dim=0)

        output_frame_count = int(output_images.shape[0])
        output_audio = self._slice_audio(extracted_audio, intervals)

        intervals_str = ", ".join([f"[{ts:.2f}s - {te:.2f}s]" for ts, te in intervals])
        audio_info_str = "None"
        if output_audio is not None:
            wf = output_audio["waveform"]
            sr = output_audio["sample_rate"]
            channels = wf.shape[1] if wf.ndim >= 2 else 1
            samples = wf.shape[-1]
            audio_dur = samples / sr
            audio_info_str = f"{channels} ch @ {sr} Hz ({samples} samples, {audio_dur:.2f}s)"

        trim_info = "\n".join([
            f"[Video Trimmer (Buff)]",
            f"Input: {total_frames} frames @ {input_fps:.2f} fps ({total_duration:.2f}s) | Size: {extracted_images.shape[2]}x{extracted_images.shape[1]}",
            f"Trim Mode: {trim_mode} | Units: {trim_units} | Windows: {intervals_str} | Duration: {trimmed_duration:.2f}s",
            f"Output: {output_frame_count} frames @ {out_fps:.2f} fps ({output_frame_count / out_fps:.2f}s) | Interpolation: {interpolation}",
            f"Audio: {audio_info_str}",
        ])

        print(trim_info)

        return (output_images, output_audio, float(out_fps), output_frame_count, trim_info)


# Node registration
NODE_CLASS_MAPPINGS = {
    "FilePathSelectorFromDirectory": FilePathSelectorFromDirectory,
    "StringProcessor": StringProcessor,
    "ConsoleOutput": ConsoleOutput,
    "TwoImageConcatenator": TwoImageConcatenator,
    "RaftOpticalFlowNode": RaftOpticalFlowNode,
    "BatchRaftOpticalFlowNode": BatchRaftOpticalFlowNode,
    "FrameRateModulator": FrameRateModulator,
    "VideoTransitionBatchMerger": VideoTransitionBatchMerger,
    "VideoFilterBatch": VideoFilterBatch,
    "VideoTrimmer": VideoTrimmer,
    "MostRecentFileSelector": MostRecentFileSelector,
    "MultilineTextSplitter": MultilineTextSplitter,
    "TextTemplateCombiner": TextTemplateCombiner,
    "LoadTextLineFromFile": LoadTextLineFromFile
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilePathSelectorFromDirectory": "File Path Selector From Directory (Buff)",
    "StringProcessor": "String Processor (Buff)",
    "ConsoleOutput": "Console Output (Buff)",
    "TwoImageConcatenator": "Two Image Concatenator (Buff)",
    "RaftOpticalFlowNode": "Raft Optical Flow Node (Buff)",
    "BatchRaftOpticalFlowNode": "Batch Raft Optical Flow Node (Buff)",
    "FrameRateModulator": "Frame Rate Modulator (Buff)",
    "VideoTransitionBatchMerger": "Video Transition Batch Merger (Buff)",
    "VideoFilterBatch": "Video Filter Batch (Buff)",
    "VideoTrimmer": "Video Trimmer (Buff)",
    "MostRecentFileSelector": "Most Recent File Selector (Buff)",
    "MultilineTextSplitter": "Multiline Text Splitter (Buff)",
    "TextTemplateCombiner": "Text Template Combiner (Buff)",
    "LoadTextLineFromFile": "Load Text Line From File (Buff)"
}


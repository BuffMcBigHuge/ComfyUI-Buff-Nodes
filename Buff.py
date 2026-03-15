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
import torch
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
                    "forceInput": True,
                    "tooltip": "When connected, this text is used instead of reading from file_path. "
                               "All line selection and filtering still applies."
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
                "weight": ("FLOAT", {
                    "default": 0.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "When non-zero, wraps output as (text:weight) for prompt weighting."
                }),
                "join_delimiter": ("STRING", {
                    "default": ", ",
                    "tooltip": "Delimiter used to join lines when selection_mode is 'all'."
                }),
                "encoding": (["utf-8", "utf-8-sig", "latin-1", "ascii", "cp1252"], {
                    "default": "utf-8",
                    "tooltip": "File encoding to use when reading the text file."
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
        text_override: str | None = None,
        skip_comments: bool = True,
        skip_blank_lines: bool = True,
        strip_whitespace: bool = True,
        replace_underscores: bool = False,
        ban_tags: str = "",
        prefix: str = "",
        suffix: str = "",
        weight: float = 0.0,
        join_delimiter: str = ", ",
        encoding: str = "utf-8",
        pick_random_file_from_dir: bool = False,
    ):
        source_file = ""

        if text_override is not None:
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


# Node registration
NODE_CLASS_MAPPINGS = {
    "FilePathSelectorFromDirectory": FilePathSelectorFromDirectory,
    "StringProcessor": StringProcessor,
    "ConsoleOutput": ConsoleOutput,
    "TwoImageConcatenator": TwoImageConcatenator,
    "RaftOpticalFlowNode": RaftOpticalFlowNode,
    "BatchRaftOpticalFlowNode": BatchRaftOpticalFlowNode,
    "MostRecentFileSelector": MostRecentFileSelector,
    "MultilineTextSplitter": MultilineTextSplitter,
    "LoadTextLineFromFile": LoadTextLineFromFile
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilePathSelectorFromDirectory": "File Path Selector From Directory (Buff)",
    "StringProcessor": "String Processor (Buff)",
    "ConsoleOutput": "Console Output (Buff)",
    "TwoImageConcatenator": "Two Image Concatenator (Buff)",
    "RaftOpticalFlowNode": "Raft Optical Flow Node (Buff)",
    "BatchRaftOpticalFlowNode": "Batch Raft Optical Flow Node (Buff)",
    "MostRecentFileSelector": "Most Recent File Selector (Buff)",
    "MultilineTextSplitter": "Multiline Text Splitter (Buff)",
    "LoadTextLineFromFile": "Load Text Line From File (Buff)"
}

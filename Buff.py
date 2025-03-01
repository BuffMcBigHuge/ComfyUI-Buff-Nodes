import os
import random
import folder_paths
import json

class FilePathSelectorFromDirectory:
    """Selects a file from a directory based on specified file types, either randomly or sequentially."""
    
    # Class variable to store indices for sequential selection
    file_indices = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": folder_paths.get_input_directory()}),
                "file_types": ("STRING", {"default": "mp4"}),
                "selection_mode": (["randomize", "sequential"], {"default": "randomize"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "select_file"
    CATEGORY = "utils"

    def select_file(self, directory_path, file_types, selection_mode, seed=None):
        # Check if directory exists
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            print(f"Error: Directory '{directory_path}' does not exist or is not a directory.")
            return ("",)
        
        # Parse file types
        extensions = [ext.strip().lower() for ext in file_types.split(",")]
        # Add dot prefix if not present
        extensions = ["." + ext if not ext.startswith(".") else ext for ext in extensions]
        
        # Get all matching files in directory
        all_files = []
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in extensions:
                    all_files.append(file_path)
        
        # Sort files for consistent sequential access and deterministic randomization
        all_files.sort()
        
        # If no files found
        if not all_files:
            print(f"No files with extensions {extensions} found in '{directory_path}'.")
            return ("",)
        
        # Create a unique key for this directory + file types combination
        dir_key = f"{directory_path}_{file_types}"
        
        if selection_mode == "randomize":
            # Seeded randomization
            if seed is not None:
                # Create a new random generator with the provided seed
                rng = random.Random(seed)
                selected_file = rng.choice(all_files)
                print(f"Selected randomized file with seed {seed}: {selected_file}")
            else:
                # Use system random if no seed provided
                selected_file = random.choice(all_files)
                print(f"Selected randomized file (unseeded): {selected_file}")
            
            return (selected_file,)
        else:
            # Sequential selection
            # Initialize index if not exists
            if dir_key not in FilePathSelectorFromDirectory.file_indices:
                FilePathSelectorFromDirectory.file_indices[dir_key] = 0
            
            # Get current index and select file
            current_index = FilePathSelectorFromDirectory.file_indices[dir_key]
            selected_file = all_files[current_index]
            
            # Update index for next time
            FilePathSelectorFromDirectory.file_indices[dir_key] = (current_index + 1) % len(all_files)
            
            print(f"Selected sequential file ({current_index + 1}/{len(all_files)}): {selected_file}")
            return (selected_file,)

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

# Node registration
NODE_CLASS_MAPPINGS = {
    "FilePathSelectorFromDirectory": FilePathSelectorFromDirectory
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilePathSelectorFromDirectory": "File Path Selector From Directory"
}

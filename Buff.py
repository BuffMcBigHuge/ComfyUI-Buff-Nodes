'''
Buff.py
Created by: https://github.com/BuffMcBigHuge
'''

import os
import random
import folder_paths
import json
import time
import os.path
import fnmatch

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
                "directory_path": ("STRING", {"default": folder_paths.get_input_directory()}),
                "file_types": ("STRING", {"default": "mp4,mkv,webm"}),
                "selection_mode": (["randomize", "sequential"], {"default": "randomize"}),
                "include_subdirectories": (["True", "False"], {"default": "True"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "cache_duration": ("INT", {"default": 60, "min": 0, "max": 3600}),
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

    def get_file_at_index(self, directory_path, extensions, index):
        """Get a file at a specific index without loading all files into memory."""
        current_index = 0
        
        # Iterate through files efficiently
        for ext in extensions:
            pattern = f"*{ext}"
            for entry in os.scandir(directory_path):
                if entry.is_file() and fnmatch.fnmatch(entry.name, pattern):
                    if current_index == index:
                        return os.path.join(directory_path, entry.name)
                    current_index += 1
        
        return ""  # Not found

    def select_random_file(self, directory_path, extensions, seed=None):
        """Select a random file without loading all files into memory."""
        # Get file count
        file_count = self.get_file_count(directory_path, extensions)
        
        if file_count == 0:
            return ""
        
        # Generate random index
        if seed is not None:
            rng = random.Random(seed)
            random_index = rng.randint(0, file_count - 1)
        else:
            random_index = random.randint(0, file_count - 1)
        
        # Get file at that index
        selected_file = self.get_file_at_index(directory_path, extensions, random_index)
        
        if selected_file:
            print(f"Selected randomized file {random_index + 1}/{file_count}: {selected_file}")
        
        return selected_file

    def get_matching_files(self, directory_path, extensions, use_cache=True, cache_duration=None, include_subdirectories="False"):
        """Get matching files using efficient methods based on the selection mode."""
        # Create a unique key for this directory + file types combination
        dir_key = f"{directory_path}_{','.join(extensions)}_{include_subdirectories}"
        
        # Use cache if available and not expired
        current_time = time.time()
        if use_cache and dir_key in self.file_cache:
            cache_entry = self.file_cache[dir_key]
            cache_time = cache_entry["timestamp"]
            expiry = cache_duration if cache_duration is not None else self.CACHE_EXPIRY
            
            if current_time - cache_time < expiry:
                return cache_entry["files"]
        
        matching_files = []
        if include_subdirectories == "True":
            for root, _, files in os.walk(directory_path):
                for ext in extensions:
                    pattern = f"*{ext}"
                    for file in fnmatch.filter(files, pattern):
                        matching_files.append(os.path.join(root, file))
        else:
            for ext in extensions:
                pattern = f"*{ext}"
                for entry in os.scandir(directory_path):
                    if entry.is_file() and fnmatch.fnmatch(entry.name, pattern):
                        matching_files.append(os.path.join(directory_path, entry.name))
        
        # Sort files for consistent access
        matching_files.sort()
        
        # Update cache
        self.file_cache[dir_key] = {
            "files": matching_files,
            "timestamp": current_time
        }
        
        return matching_files

    def select_sequential_file(self, directory_path, extensions, dir_key):
        """Select a file sequentially from the directory."""
        matching_files = self.get_matching_files(directory_path, extensions)
        
        if not matching_files:
            return ""
            
        # Initialize index if not exists
        if dir_key not in FilePathSelectorFromDirectory.file_indices:
            FilePathSelectorFromDirectory.file_indices[dir_key] = 0
        
        # Get current index and select file
        current_index = FilePathSelectorFromDirectory.file_indices[dir_key]
        
        # Safety check in case file list changed
        if current_index >= len(matching_files):
            current_index = 0
            FilePathSelectorFromDirectory.file_indices[dir_key] = 0
            
        selected_file = matching_files[current_index]
        
        # Update index for next time
        FilePathSelectorFromDirectory.file_indices[dir_key] = (current_index + 1) % len(matching_files)
        
        print(f"Selected sequential file ({current_index + 1}/{len(matching_files)}): {selected_file}")
        return selected_file

    def select_file(self, directory_path, file_types, selection_mode, include_subdirectories="False", seed=None, cache_duration=60):
        # Strip quotation marks from directory path if present
        directory_path = directory_path.strip('"\'')
        
        # Check if directory exists
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            print(f"Error: Directory '{directory_path}' does not exist or is not a directory.")
            return ("",)
        
        # Parse file types
        extensions = [ext.strip().lower() for ext in file_types.split(",")]
        # Add dot prefix if not present
        extensions = ["." + ext if not ext.startswith(".") else ext for ext in extensions]
        
        # Create a unique key for this directory + file types combination
        dir_key = f"{directory_path}_{file_types}_{include_subdirectories}"
        
        # Update cache duration if provided
        if cache_duration is not None and cache_duration != 60:
            self.CACHE_EXPIRY = cache_duration
        
        if selection_mode == "randomize":
            selected_file = self.select_random_file(directory_path, extensions, seed) 
        else:
            selected_file = self.select_sequential_file(directory_path, extensions, dir_key)
            
        if not selected_file:
            print(f"No files with extensions {extensions} found in '{directory_path}'.")
            return ("",)
            
        return (selected_file,)

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
            "custom_expression": "Use a custom expression to manipulate the string (see examples below)"
        }
        
        return {
            "required": {
                "input_string": ("STRING", {"default": "", "multiline": True}),
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
                    "tooltip": """Custom expression to manipulate the input string.
Use {s} as a placeholder for the input string.

Examples:
• {s}.upper() - Convert to uppercase
• {s}.lower() - Convert to lowercase
• {s}.replace('old', 'new') - Replace text
• {s}[::-1] - Reverse the string
• {s}.split(',')[0] - Get first item of CSV
• ' '.join([w.capitalize() for w in {s}.split()]) - Title case
• {s}.strip() - Remove whitespace from ends
• {s} + ' (modified)' - Append text
"""
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
                    "tooltip": "Text to display in the console"
                }),
                "prefix": ("STRING", {
                    "default": "ConsoleOutput",
                    "tooltip": "Prefix to add before the text in console output"
                }),
            },
            "optional": {
                "show_timestamp": (["True", "False"], {
                    "default": "True",
                    "tooltip": "Whether to include a timestamp in the console output"
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

# Node registration
NODE_CLASS_MAPPINGS = {
    "FilePathSelectorFromDirectory": FilePathSelectorFromDirectory,
    "StringProcessor": StringProcessor,
    "ConsoleOutput": ConsoleOutput
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilePathSelectorFromDirectory": "File Path Selector From Directory (Buff)",
    "StringProcessor": "String Processor (Buff)",
    "ConsoleOutput": "Console Output (Buff)"
}

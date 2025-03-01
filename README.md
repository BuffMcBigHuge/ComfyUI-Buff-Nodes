# ComfyUI-Buff-Nodes

_WORK IN PROGRESS_

## File Path Selector Node

The `FilePathSelectorFromDirectory` node is a utility for ComfyUI that helps you select files from a specified directory. 
- **Directory Selection**: Choose any directory on your system to scan for files
- **File Type Filtering**: Specify which file extensions to include (e.g., "mp4,jpg,png")
- **Selection Modes**:
  - **Randomize**: Pick a random file from the directory (can be seeded for reproducibility)
  - **Sequential**: Go through files one by one in alphabetical order, remembering position between runs
- **Persistent Memory**: Tracks the sequential position between ComfyUI sessions

## Use Cases

- Randomly selecting input videos or images for processing
- Creating workflows that process batches of files in sequence
- Testing workflows with different inputs without manual file selection

## Technical Details

The node saves its sequential position state to a JSON file, allowing it to remember where it left off even if you restart ComfyUI. When using the randomize mode with a seed, you'll get consistent, reproducible selections.
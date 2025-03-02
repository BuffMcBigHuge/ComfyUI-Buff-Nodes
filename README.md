# ComfyUI-Buff-Nodes

Several quality-of-life batch operation and string manipulation nodes.

_WORK IN PROGRESS_

In part written with Claude-3.7-Sonnet in a human in-the-loop development standard.

## File Path Selector Node

The `FilePathSelectorFromDirectory` node is a utility for ComfyUI that helps you select files from a specified directory. 
- **Directory Selection**: Choose any directory on your system to scan for files
- **File Type Filtering**: Specify which file extensions to include (e.g., "mp4,jpg,png")
- **Selection Modes**:
  - **Randomize**: Pick a random file from the directory (can be seeded for reproducibility)
  - **Sequential**: Go through files one by one in alphabetical order, remembering position between runs
- **Persistent Memory**: Tracks the sequential position between ComfyUI sessions

## String Processor Node

The `StringProcessor` node allows you to manipulate text strings with various operations:
- **Operations**:
  - **First/Last N Characters**: Extract the first or last specified number of characters
  - **First/Last N Words**: Extract the first or last specified number of words using a custom delimiter
  - **Custom Expression**: Use Python expressions to transform strings with full flexibility
- **Features**:
  - Custom word delimiters
  - Comprehensive tooltips with examples
  - Error handling for custom expressions

## Console Output Node

The `ConsoleOutput` node provides a simple way to display text in the console:
- **Features**:
  - Custom prefix to identify your outputs
  - Optional timestamps
  - Acts as a terminal output node in workflows
  - Useful for debugging or logging information

## Use Cases

- Randomly selecting input files for processing
- Creating workflows that process batches of files in sequence
- Manipulating text data within workflows (filenames, prompts, metadata)
- Extracting portions of text from larger strings
- Debugging workflows by logging intermediate values to the console
- Formatting and transforming text for use in prompts or file operations

## Technical Details

The File Path Selector node saves its sequential position state to a JSON file, allowing it to remember where it left off even if you restart ComfyUI. When using the randomize mode with a seed, you'll get consistent, reproducible selections.

The String Processor supports Python expressions that give you the full power of string manipulation in a controlled environment.
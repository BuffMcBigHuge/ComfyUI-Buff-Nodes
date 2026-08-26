# ComfyUI-Buff-Nodes

Several quality-of-life batch operation and string manipulation nodes.

## File Path Selector Node

The `FilePathSelectorFromDirectory` node is a utility for ComfyUI that helps you select files from a specified directory. 
- **Directory Selection**: Choose any directory on your system to scan for files
- **File Type Filtering**: Specify which file extensions to include (e.g., "mp4,jpg,png")
- **Selection Modes**:
  - **Randomize**: Pick a random file from the directory (can be seeded for reproducibility)
  - **Sequential**: Go through files one by one in alphabetical order, remembering position between runs
- **Subdirectory Support**: Optionally search through subdirectories for matching files
- **Persistent Memory**: Tracks the sequential position between ComfyUI sessions
- **Caching**: Efficient file list caching with configurable duration

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

## Multiline Text Splitter Node

The `MultilineTextSplitter` node splits multiline text into multiple output fields:
- **Features**:
  - Split input text across 1–20 output fields
  - **Split Modes**:
    - **Even Split**: Distributes lines evenly across outputs
    - **Fixed Lines**: Assigns a fixed number of lines per output (configurable via `lines_per_output`)
  - Optional removal of empty lines before processing
- **Input Options**:
  - **Input Text**: Multiline text to split
  - **Num Outputs**: Number of output fields (1–20)
  - **Lines Per Output**: Lines per output when not splitting evenly
  - **Split Evenly**: When True, splits evenly; when False, uses `lines_per_output` sequentially
  - **Remove Empty Lines**: Optionally strip empty lines before splitting
- **Output**: Up to 20 text outputs (`text_1` through `text_20`); unused outputs return empty strings

## Text Template Combiner Node

The `TextTemplateCombiner` node combines connected string inputs into one output using numbered placeholders in a primary text box:
- **Template Replacement**:
  - Write the base text in `main`
  - Use `{1}`, `{2}`, etc. where connected strings should be inserted
  - Example: `main = "foo bar {1}"`, `text_1 = "inserted string"` outputs `foo bar inserted string`
- **Random Choices**:
  - Use `{foo|bar}` to randomly output either `foo` or `bar`
  - Pipe groups support more than two options, such as `{red|green|blue}`
  - Numeric options can reference dynamic inputs: `{1|2}` randomly chooses `text_1` or `text_2`
- **Dynamic Inputs**:
  - Set `inputcount`, then click `Update text inputs` to show exactly that many `text_N` sockets
  - Adding placeholders such as `{4}` or choice groups such as `{1|4}` in `main` also expands the visible sockets up to `text_4`
  - `text_1` through `text_20` are declared on the backend for API-safe workflows
- **Output**:
  - Single combined `STRING` output

## Load Text Line From File Node

The `LoadTextLineFromFile` node loads text from a file and returns a specific line or all lines:
- **Features**:
  - Supports `.txt` and `.csv` files
  - **Selection Modes**:
    - **Random**: Pick a random line (seeded for reproducibility)
    - **Index**: Pick a specific line by index (wraps around if out of range)
    - **Sequential**: Cycle through lines in order across executions
    - **All**: Return every line joined by a delimiter
  - **Text Override**: Optional input to use connected text instead of reading from file (filtering still applies)
- **Filtering Options**:
  - Skip lines starting with `#` (comments)
  - Skip blank lines
  - Strip leading/trailing whitespace
  - Replace underscores with spaces
  - Remove specified tags via comma-separated `ban_tags` list
- **Output Formatting**:
  - Prefix and suffix
  - Prompt weighting: wrap output as `(text:weight)` when weight is non-zero
  - Custom join delimiter for "all" mode
- **Directory Support**: When `file_path` is a directory and `pick_random_file_from_dir` is True, selects a random `.txt` file from it
- **Encoding**: Configurable file encoding (utf-8, utf-8-sig, latin-1, ascii, cp1252)
- **Output**: Returns `(text, source_file, line_count)`

## Console Output Node

The `ConsoleOutput` node provides a simple way to display text in the console:
- **Features**:
  - Custom prefix to identify your outputs
  - Optional timestamps
  - Acts as a terminal output node in workflows
  - Useful for debugging or logging information

## Most Recent File Selector Node

The `MostRecentFileSelector` node finds the most recently modified file in a specified directory:
- **Features**:
  - Automatically detects the most recently modified file in a directory
  - Optional file type filtering using comma-separated extensions
  - Can create a black initialization image if no file is found
  - Configurable black image dimensions and format
- **Input Options**:
  - **Directory Path**: Path to search for files (defaults to ComfyUI's input directory)
  - **File Types**: Optional comma-separated list of file extensions to filter by (e.g., "png,jpg")
  - **Create Init Image**: Option to create a black image if no file is found
  - **Init Image Settings**: Configurable width, height, and format (png/jpg) for the black image
- **Output**:
  - Returns the full path to the most recent file
  - Returns an empty string if no file is found and init image creation is disabled
  - Creates and returns path to a black image if no file is found and init image creation is enabled
- **Behavior**:
  - Always re-executes when the workflow is queued
  - Handles missing directories gracefully
  - Provides detailed console logging for debugging

## Two Image Concatenator Node

> Note: The following node is for the use of [TemporalNet2](https://huggingface.co/CiaraRowles/TemporalNet2) in video streaming workflows

The `TwoImageConcatenator` node combines two 3-channel images into a single 6-channel image:
- **Features**:
  - Concatenates two input images along the channel dimension
  - Maintains batch processing capability
  - Useful for temporal models that require both previous and current frames
  - Preserves image dimensions while doubling the channel count
- **Input Requirements**:
  - Both images must have the same batch size, height, and width
  - Both images must be 3-channel images
- **Output**:
  - Single 6-channel image tensor (batch, height, width, 6 channels)

## Frame Rate Modulator Node

The `FrameRateModulator` node allows you to modify the frame rate of image sequences by resampling frames:
- **Features**:
  - **Mode Selection**: Choose between exact frame count or multiplier-based resampling
  - **Frame Count Mode**: Set a specific target number of output frames
  - **Multiplier Mode**: Scale input frame count by a factor (2.0 = slow motion, 0.5 = speed up)
  - **Interpolation Methods**:
    - **Nearest**: Simple frame duplication/skipping (fastest)
    - **Linear**: Smooth blending between adjacent frames
    - **Cubic**: High-quality cubic interpolation for smooth motion
  - **Loop Modes** for handling edge cases:
    - **Clamp**: Use first/last frame for out-of-bounds positions
    - **Repeat**: Loop the sequence seamlessly
    - **Mirror**: Reverse and repeat for ping-pong effect
- **Input Requirements**:
  - Batch of images in ComfyUI format (batch, height, width, channels)
- **Output**:
  - Resampled image sequence with target frame count
  - Frame count as integer for downstream processing
- **Use Cases**:
  - Creating slow motion effects (50 frames → 75 frames)
  - Speeding up sequences (50 frames → 25 frames)
  - Standardizing frame counts across different clips
  - Temporal upsampling with smooth interpolation

## Video Transition Batch Merger Node

The `VideoTransitionBatchMerger` node merges multiple ComfyUI image batches into a single output timeline with configurable transitions between clips:
- **Multi-Clip Timeline**:
  - Connect `image_batch_1` and `image_batch_2` for a standard A/B transition
  - Connect optional `audio_1` and `audio_2` when clips have audio; ComfyUI `AUDIO`, lazy `AUDIO` from Video Helper Suite `Load Video` nodes, and legacy `VHS_AUDIO` are accepted
  - Set `inputcount`, then click `Update inputs/transitions` to add or remove `image_batch_3+` and `audio_3+` sockets
  - `image_batch_3` through `image_batch_64` and matching `audio_N` sockets are declared on the backend, so raw Prompt API workflows can connect them without relying on the browser UI
  - The same update button adds or removes `transition_N` and `transition_frames_N` controls for each clip boundary
  - `transition_plan` stores those dynamic pair controls for the backend and supports manual overrides such as `8:dip_to_white:10`
- **Audio Timeline**:
  - `fps` controls the full output duration and audio alignment
  - Clips without audio produce silence for their portion of the timeline
  - Non-overlapping inserted transition frames produce silence
  - Overlapping transitions trim audio to match the shortened video timeline and crossfade the overlapping audio handles
- **Overlapping A/B Transitions**:
  - Cross dissolve
  - Additive / film dissolve
  - Linear wipes in horizontal, vertical, and diagonal directions
  - Barn doors open/close
  - Radial/clock wipe
  - Iris circle, diamond, and star wipes
  - Push and slide transitions in all four directions
  - Zoom, dissolve zoom, checkerboard, band slide, glitch, film burn, spin, page peel, and cube spin
- **Non-Overlapping Inserted Transitions**:
  - Dip to black
  - Dip to white
  - Morph-cut style generated in-between frames
  - Whip pan / swish pan in horizontal or vertical directions
  - Light leak / flare transition
- **Timeline Modes**:
  - `auto`: uses overlap for A/B roll transitions and inserted frames for dip/camera-artifact transitions
  - `overlap`: consumes handles from the outgoing and incoming clips, so output frames are less than the total input frames
  - `non_overlap`: inserts generated transition frames between clips, so output frames are greater than or equal to the total input frames
- **Output**:
  - Merged image batch
  - Single aligned audio track
  - FPS passthrough
  - Final frame count
  - Timeline summary string for debugging

## Video Filter Batch Node

The `VideoFilterBatch` node applies tensor-based color presets and video effects to a ComfyUI image batch. It runs in the backend, so it works through the ComfyUI API without browser-only dependencies.
- **Filter Presets**:
  - Includes 22 looks: Sunset Glow, Moonlight, Cyberpunk, Film Noir, Dreamy, Retro VHS, Teal & Orange, Bleach Bypass, Cross Process, Faded Film, Warm/Cool Cinematic, Vintage Chrome, Golden Hour, Sepia, Kodak Portra, Fuji Velvia, Polaroid, Horror Green, Arctic Blue, Desert Heat, and Matrix Green
  - `preset_strength` controls the intensity of the selected look
- **Effects**:
  - Includes Sharpen, Chromatic Aberration, Vignette, Film Grain, Glitch, RGB Split, Lens Distortion, Motion Blur, Posterize, Duotone, Bloom, and Scanlines
  - Three dropdown effect slots are available for common UI use
  - `effect_stack` accepts additional API-friendly lines such as `film_grain:0.35` or `chromatic-aberration:0.5`
  - `seed` makes grain and glitch effects deterministic
  - Motion Blur is temporal and motion-masked from adjacent frames, so static frames stay mostly unchanged
- **Output**:
  - Filtered image batch
  - Filter summary string for debugging

## Video Trimmer Node

The `VideoTrimmer` node allows you to perform versatile video trimming and framerate modulation with sample-accurate audio synchronization (preventing audio skew or pitch drift):
- **Inputs**:
  - **Native Video**: Connect native ComfyUI `VIDEO` objects (from `LoadVideo`, `CreateVideo`, etc.) with automatic framerate detection and audio extraction
  - **Image Batches**: Connect ComfyUI `IMAGE` batches `[B, H, W, C]`
  - **Audio**: Optional ComfyUI `AUDIO` dict or `VHS_AUDIO`
- **Trimming Modes (`trim_mode`)**:
  - `all`: Retains the entire clip (ideal for framerate conversion / resampling)
  - `start_duration`: Extracts a segment from `start_value` with duration/length `duration_or_count` (supports negative start to index from the end)
  - `start_end`: Extracts a segment between `start_value` and `end_value` (supports negative end indexing)
  - `first_n`: Extracts the first N units (frames or seconds)
  - `last_n`: Extracts the last N units (frames or seconds)
  - `center_n`: Extracts N units centered around the midpoint of the video
  - `cut_inside`: Removes a section between `start_value` and `end_value`, seamlessly splicing the preceding and succeeding segments and their audio
- **Flexible Units (`trim_units`)**:
  - `seconds`: Time-based trimming in real-world seconds
  - `source_frames`: Trim by frame count / indices based on source framerate (e.g. extract last 48 frames of a 24 fps video)
  - `output_frames`: Trim by frame count based on target framerate (e.g. extract last 24 frames when target fps is 12)
  - `percentage`: Trim by percentage (0% to 100% of duration)
- **Framerate Modulation & Audio Sync (`framerate_mode`)**:
  - `keep_source`: Retains original framerate
  - `custom_fps`: Resamples video frames to a specific target framerate (e.g. 12 fps, 24 fps, 60 fps) across the real-world trimmed duration. Because real playback time matches audio duration, **audio remains 100% in sync without pitch skew or speed warping**
  - `multiplier`: Scales source framerate by a multiplier (0.5x, 2.0x, etc.)
  - `target_frame_count`: Forces the output to have an exact number of frames, automatically deriving the matching output framerate
- **Interpolation Methods**:
  - `nearest`: Nearest-neighbor frame skipping/duplication (fast, crisp, no ghosting/blur)
  - `linear`: Smooth linear blending between neighboring frames
  - `cubic`: Smooth Catmull-Rom cubic interpolation
- **Outputs**:
  - `images`: Trimmed/resampled image batch
  - `audio`: Trimmed, sample-accurate audio track (or None if no audio)
  - `fps`: Effective output framerate (FLOAT)
  - `frame_count`: Output frame count (INT)
  - `trim_info`: Detailed summary of input/output frame counts, framerates, durations, and audio parameters


## Raft Optical Flow Node

> Note: The following node is for the use of [TemporalNet2](https://huggingface.co/CiaraRowles/TemporalNet2) in video streaming workflows

The `RaftOpticalFlowNode` calculates optical flow between two images using the RAFT (Recurrent All-Pairs Field Transforms) model:
- **Features**:
  - Uses the state-of-the-art RAFT model for optical flow estimation
  - Processes images in batches
  - Automatically handles device placement (CPU/GPU)
  - Produces a visualization of motion between frames
- **Input Requirements**:
  - Two input images (typically previous and current frames)
  - Images should be in ComfyUI's standard format (batch, height, width, channels)
- **Output**:
  - Optical flow visualization as an image
  - Color-coded representation of motion between frames
  - Output is normalized to [0,1] range

## Use Cases

- Randomly selecting input files for processing
- Creating workflows that process batches of files in sequence
- Manipulating text data within workflows (filenames, prompts, metadata)
- Extracting portions of text from larger strings
- Splitting multiline prompts or lists across multiple nodes (e.g., batch prompts)
- Loading prompts, tags, or captions from text files with random or sequential selection
- Building prompt libraries from .txt or .csv files with filtering and weighting
- Debugging workflows by logging intermediate values to the console
- Formatting and transforming text for use in prompts or file operations
- Processing video frames for motion analysis
- Combining multiple image channels for temporal models
- Visualizing motion between consecutive frames
- Finding the latest output from a previous workflow run
- Creating initialization images for new workflows
- Monitoring directories for new files
- Setting up automated workflows that process the most recent file

## Technical Details

The File Path Selector node saves its sequential position state to a JSON file, allowing it to remember where it left off even if you restart ComfyUI. When using the randomize mode with a seed, you'll get consistent, reproducible selections.

The String Processor supports Python expressions that give you the full power of string manipulation in a controlled environment.

The Raft Optical Flow Node uses the RAFT model from torchvision, which provides high-quality optical flow estimation. The model is automatically downloaded when first used and cached for subsequent runs.

from .Buff import FilePathSelectorFromDirectory, ConsoleOutput, StringProcessor, TwoImageConcatenator, MostRecentFileSelector, RaftOpticalFlowNode

NODE_CLASS_MAPPINGS = {
    "FilePathSelectorFromDirectory": FilePathSelectorFromDirectory,
    "StringProcessor": StringProcessor,
    "ConsoleOutput": ConsoleOutput,
    "TwoImageConcatenator": TwoImageConcatenator,
    "MostRecentFileSelector": MostRecentFileSelector,
    "RaftOpticalFlowNode": RaftOpticalFlowNode,
    "MostRecentFileSelector": MostRecentFileSelector,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FilePathSelectorFromDirectory": "File Path Selector From Directory (Buff)",
    "StringProcessor": "String Processor (Buff)",
    "ConsoleOutput": "Console Output (Buff)",
    "TwoImageConcatenator": "Two Image Concatenator (Buff)",
    "RaftOpticalFlowNode": "Raft Optical Flow Node (Buff)",
    "MostRecentFileSelector": "Most Recent File Selector (Buff)",
}

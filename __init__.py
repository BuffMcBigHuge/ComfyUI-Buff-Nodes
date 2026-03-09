from .Buff import FilePathSelectorFromDirectory, ConsoleOutput, StringProcessor, TwoImageConcatenator, MostRecentFileSelector, RaftOpticalFlowNode, FrameRateModulator, MultilineTextSplitter, LoadTextLineFromFile

NODE_CLASS_MAPPINGS = {
    "FilePathSelectorFromDirectory": FilePathSelectorFromDirectory,
    "StringProcessor": StringProcessor,
    "ConsoleOutput": ConsoleOutput,
    "TwoImageConcatenator": TwoImageConcatenator,
    "MostRecentFileSelector": MostRecentFileSelector,
    "RaftOpticalFlowNode": RaftOpticalFlowNode,
    "FrameRateModulator": FrameRateModulator,
    "MultilineTextSplitter": MultilineTextSplitter,
    "LoadTextLineFromFile": LoadTextLineFromFile,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FilePathSelectorFromDirectory": "File Path Selector From Directory (Buff)",
    "StringProcessor": "String Processor (Buff)",
    "ConsoleOutput": "Console Output (Buff)",
    "TwoImageConcatenator": "Two Image Concatenator (Buff)",
    "RaftOpticalFlowNode": "Raft Optical Flow Node (Buff)",
    "MostRecentFileSelector": "Most Recent File Selector (Buff)",
    "FrameRateModulator": "Frame Rate Modulator (Buff)",
    "MultilineTextSplitter": "Multiline Text Splitter (Buff)",
    "LoadTextLineFromFile": "Load Text Line From File (Buff)",
}

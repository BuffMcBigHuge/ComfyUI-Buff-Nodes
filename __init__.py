from .Buff import FilePathSelectorFromDirectory, ConsoleOutput, StringProcessor, TwoImageConcatenator, MostRecentFileSelector, RaftOpticalFlowNode, BatchRaftOpticalFlowNode, FrameRateModulator, VideoTransitionBatchMerger, VideoFilterBatch, VideoTrimmer, MultilineTextSplitter, TextTemplateCombiner, LoadTextLineFromFile

NODE_CLASS_MAPPINGS = {
    "FilePathSelectorFromDirectory": FilePathSelectorFromDirectory,
    "StringProcessor": StringProcessor,
    "ConsoleOutput": ConsoleOutput,
    "TwoImageConcatenator": TwoImageConcatenator,
    "MostRecentFileSelector": MostRecentFileSelector,
    "RaftOpticalFlowNode": RaftOpticalFlowNode,
    "BatchRaftOpticalFlowNode": BatchRaftOpticalFlowNode,
    "FrameRateModulator": FrameRateModulator,
    "VideoTransitionBatchMerger": VideoTransitionBatchMerger,
    "VideoFilterBatch": VideoFilterBatch,
    "VideoTrimmer": VideoTrimmer,
    "MultilineTextSplitter": MultilineTextSplitter,
    "TextTemplateCombiner": TextTemplateCombiner,
    "LoadTextLineFromFile": LoadTextLineFromFile,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FilePathSelectorFromDirectory": "File Path Selector From Directory (Buff)",
    "StringProcessor": "String Processor (Buff)",
    "ConsoleOutput": "Console Output (Buff)",
    "TwoImageConcatenator": "Two Image Concatenator (Buff)",
    "RaftOpticalFlowNode": "Raft Optical Flow Node (Buff)",
    "BatchRaftOpticalFlowNode": "Batch Raft Optical Flow Node (Buff)",
    "MostRecentFileSelector": "Most Recent File Selector (Buff)",
    "FrameRateModulator": "Frame Rate Modulator (Buff)",
    "VideoTransitionBatchMerger": "Video Transition Batch Merger (Buff)",
    "VideoFilterBatch": "Video Filter Batch (Buff)",
    "VideoTrimmer": "Video Trimmer (Buff)",
    "MultilineTextSplitter": "Multiline Text Splitter (Buff)",
    "TextTemplateCombiner": "Text Template Combiner (Buff)",
    "LoadTextLineFromFile": "Load Text Line From File (Buff)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


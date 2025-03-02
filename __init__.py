from .Buff import FilePathSelectorFromDirectory, ConsoleOutput, StringProcessor

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

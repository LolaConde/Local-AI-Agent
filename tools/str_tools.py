#imports
from llama_index.core.tools import FunctionTool

def reverse_string(s: str) -> str:
    """
    Reverses the input string.

    Args:
        s (str): The string to be reversed.

    Returns:
        str: The reversed string.
    """
    return s[::-1]

reverse_string_tool = FunctionTool.from_defaults(
    fn=reverse_string,
    name="reverse_string",
    description=(
        "Reverses the input string. "
        "Takes a string as input and returns the reversed string."
    ),
)
import pandas as pd
from llama_index.core.tools import FunctionTool

def read_file(file_path):
    "Function that reads a file and returns its content as a string."
    # It is assumed that the file is located in the 'files' directory relative to the current working directory.
    file_path = f"./files/{file_path}"
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
        return df.to_string(index=False)
    elif file_path.endswith('.py'):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        return "I cannot read this file type."

read_file_tool = FunctionTool.from_defaults(
    fn=read_file,
    name="read_file",
    description=(
        "Reads a file and returns its content as a string. "
        "Supports .xlsx and .py files."
    ),
)
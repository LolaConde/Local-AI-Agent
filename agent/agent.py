# Models imports
from models.models_factory import LlamaIndexModelFactory
# Agent imports (llamaindex)
from llama_index.core.agent.workflow import AgentWorkflow, ToolCallResult, AgentStream
# Tools imports
from tools.files_tools import read_file_tool
from tools.web_tools import web_search_tool, web_rag_tool, web_tables_tool
from tools.calculator_tools import calculator_tool
from tools.str_tools import reverse_string_tool
from tools.audio_tool import transcribe_audio_tool

# The prompt is a modified version from the GAIA documentation, adding or changing some phrases that have been seen to be good to clarify
SYSTEM_PROMPT = """
You are a general AI assistant.

Use calculator for all arithmetic, including calculations on tool results.
Use search tools for required facts not supplied by the user or obtained from tools.

After using the tools, if the information is sufficient to answer the question, answer it. If it is, answer the question. If not, state what is missing and use other tools if needed. Use at much 3 searches per question so do only important searches.

Keep tool queries focused on the original question.
Do not repeat calls that return no new information.
Retrieval is not scrolling: change the source or method if needed.

When the evidence is sufficient, answer. Otherwise, state what is missing.

Finish with: FINAL ANSWER: [YOUR FINAL ANSWER]
Do not change the format of the final answer.
Keep the final answer as short as possible.
For numbers, omit thousands separators and units unless requested, and follow the requested decimal precision.
For names, omit unnecessary articles and do not abbreviate.
For comma-separated lists, apply these rules to each item. Also, comma-separated lists should contain spaces.
"""

import asyncio
from collections import Counter

from llama_index.core.tools import FunctionTool


def limit_tools(tools, max_calls=3):
    counts = Counter()
    lock = asyncio.Lock()

    def wrap(tool):
        name = tool.metadata.name

        async def limited_call(**kwargs):
            async with lock:
                if counts[name] >= max_calls:
                    return (
                        f"TOOL_LIMIT_REACHED: {name} has reached its "
                        f"limit of {max_calls} executions for this question. "
                        "Do not call it again. Use another method or "
                        "answer with the available evidence."
                    )

                # Increment the count for this tool
                counts[name] += 1

            return await tool.acall(**kwargs)

        return FunctionTool.from_defaults(
            async_fn=limited_call,
            tool_metadata=tool.metadata,
        )

    return [wrap(tool) for tool in tools]

class BasicAgent:
    PRINT_LLM_OUTPUT = False
    def __init__(self):
        if self.PRINT_LLM_OUTPUT:
            print("BasicAgent initialized.")
        llm = LlamaIndexModelFactory().obtain_LLM_model()

        tools = [
            calculator_tool,
            web_search_tool,
            web_rag_tool,
            web_tables_tool,
            reverse_string_tool,
            read_file_tool,
            transcribe_audio_tool
        ]

        tools = limit_tools(tools, max_calls=3)

        self.agent = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=tools,
            llm=llm,
            system_prompt=SYSTEM_PROMPT,
            timeout=180.0,
        )

    def get_answer(self, answer:str) -> str:
        """Extracts the final answer from the agent's response."""
        final_answer_prefix = "FINAL ANSWER:"
        if final_answer_prefix in answer:
            return_str = answer.split(final_answer_prefix)[-1].strip()
        else:
            return_str = answer

        # If the answer is separated by commas, put spaces after the commas if not already present
        if "," in return_str:
            return_str = ", ".join(part.strip() for part in return_str.split(","))

        # If not commas (not a list), start with capital letter the first letter of the first word
        else:
            return_str = return_str[0].upper() + return_str[1:] if return_str else return_str

        return return_str
    async def __call__(self, question: str) -> str:
            if self.PRINT_LLM_OUTPUT:
                print(f"Agent received question (first 50 chars): {question[:50]}...")
            handler = self.agent.run(question)
            if self.PRINT_LLM_OUTPUT:
                async for ev in handler.stream_events():
                    if isinstance(ev, ToolCallResult):
                        print("")
                        print("Called tool: ", ev.tool_name, ev.tool_kwargs, "=>", ev.tool_output)
                    elif isinstance(ev, AgentStream):  # showing the thought process
                        print(ev.delta, end="", flush=True)
                    else:
                        print("Other event: ")
                        print(ev, end="", flush=True)
                print("")  # Print a newline after the thought process
            response = await handler
            return response
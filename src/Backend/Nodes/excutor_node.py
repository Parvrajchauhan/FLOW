from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from src.Backend.LLMs.geminiLLM import llm
from src.Backend.state.State import AgentState
from src.Backend.tools.extract_tool import extract_tool
from src.Backend.tools.audio_tool import audio_tool
from src.Backend.tools.YouTube_tool import youtube_tool
import json

tools = [extract_tool, audio_tool, youtube_tool]
executor_llm = llm.bind_tools(tools)

EXECUTOR_SYSTEM = """
You are a task executor. Your ONLY job is to call tools in the exact sequence given.
Do not reason, summarize, or answer the user — only call tools.

Execute this tool sequence in order: {tool_sequence}

File registry (use content_b64 and file_type from here when calling extract_tool or audio_tool):
{file_registry}

Rules:
- Call tools one at a time in the order given.
- Pass the correct args from file_registry to each tool.
- For youtube_tool: extract the YouTube URL from the user query or extracted text.
- Stop after all tools in the sequence are called.
"""


def executor_node(state: AgentState) -> AgentState:
    tool_sequence = state.get("tool_sequence", [])

    # nothing to execute — skip straight through
    if not tool_sequence:
        return state

    system = EXECUTOR_SYSTEM.format(
        tool_sequence=tool_sequence,
        file_registry=json.dumps(state.get("file_registry", {})),
    )

    messages = [SystemMessage(content=system)] + state["messages"]

    response = executor_llm.invoke(messages)

    return {
        **state,
        "messages": [response],  # AIMessage with tool_calls
    }
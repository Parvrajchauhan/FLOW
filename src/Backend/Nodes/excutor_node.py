import json
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage

from src.Backend.LLMs.geminiLLM import get_llm
from src.Backend.state.State import State
from src.Backend.tools.extract_tool import extract_tool
from src.Backend.tools.audio_tool import audio_tool
from src.Backend.tools.YouTube_tool import youtube_tool

tools = [extract_tool, audio_tool, youtube_tool]
executor_llm = get_llm().bind_tools(tools)

SYSTEM = """You are a tool execution node. Your sole responsibility is to make exactly the tool call specified in `tool_call`.
Context
- Tool to call: {tool_call}
- File registry: {file_registry}
- Extracted texts: {extracted_texts}
- Audio transcript: {audio_transcript}

Rules
1. Make ONLY the tool call defined in `tool_call`. No other tool calls.
2. Do NOT summarize, analyze, explain, or produce any text output.
3. Do NOT make additional tool calls beyond the one specified.

YouTube extraction
If a YouTube URL appears in the user query or extracted texts AND `tool_call` references `YouTube_tool`, 
extract the URL from the context and pass it to `YouTube_tool`.

Routing
- If `tool_call` is populated → execute it.
- If `tool_call` is empty → skip execution and go directly to planner.
"""


def get_filename_from_path(file_registry: dict, file_path: str) -> str | None:
    for filename, meta in file_registry.items():
        if meta.get("path") == file_path:
            return filename
    return None


def parse_tool_results(state: State) -> dict:
    extracted_texts = dict(state.get("extracted_texts", {}))
    audio_transcript = state.get("audio_transcript", "")
    yt_transcript = state.get("yt_transcript", "")
    ocr_confidences = state.get("ocr_confidences", {})

    messages = state.get("messages", [])
    if not messages: return {}

    last_tool = messages[-1]
    if not isinstance(last_tool, ToolMessage): return {}

    try:
        data = json.loads(last_tool.content)
    except Exception: return {}

    tool_call_id = last_tool.tool_call_id
    tool_path = None

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        for tc in getattr(msg, "tool_calls", []):
            if tc["id"] == tool_call_id:
                tool_path = tc["args"].get("file_path")
                break
        if tool_path:
            break

    # extract_tool
    if last_tool.name == "extract_tool" and "extracted_text" in data:
        filename = get_filename_from_path(state.get("file_registry", {}), tool_path)
        if filename:
            extracted_texts[filename] = data["extracted_text"]
            ocr_confidences[filename] = data.get("ocr_confidence", {})

    # audio_tool
    elif last_tool.name == "audio_tool" and "transcript" in data:
        audio_transcript = data["transcript"]

    # youtube_tool
    elif last_tool.name == "youtube_tool" and "transcript" in data:
        yt_transcript = data["transcript"]

    return {
        "extracted_texts": extracted_texts,
        "ocr_confidences": ocr_confidences,
        "audio_transcript": audio_transcript,
        "yt_transcript": yt_transcript,
    }


async def executor_node(state: State) -> dict:
    updates = parse_tool_results(state)

    tool_call = str(state.get("tool_call", "")).strip().lower()

    if tool_call in ["", "none", "null"]:
        return {}

    all_messages = state["messages"]

    if all_messages and isinstance(all_messages[-1], ToolMessage):
        return {
            **updates,
            "plan_trace": ["Executor_Node: Tool completed, routing back to planner"],
        }

    system = SYSTEM.format(
        tool_call=tool_call,
        file_registry=json.dumps(state.get("file_registry", {}), indent=2),
        extracted_texts=json.dumps(state.get("extracted_texts", {}), indent=2),
        audio_transcript=state.get("audio_transcript", ""),
    )

    messages_to_send = [SystemMessage(content=system),HumanMessage(content=f"Execute tool: {tool_call}")]
    response = await executor_llm.ainvoke(messages_to_send)

    return {
        "messages": [response],
        "plan_trace": ["Executor_Node: Tool call dispatched"],
        **updates,
    }

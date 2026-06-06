import json

from langchain_core.messages import (
    SystemMessage,
    ToolMessage,
    AIMessage,
)

from src.Backend.LLMs.geminiLLM import get_llm
from src.Backend.state.State import State
from src.Backend.tools.extract_tool import extract_tool
from src.Backend.tools.audio_tool import audio_tool
from src.Backend.tools.YouTube_tool import youtube_tool

tools = [extract_tool, audio_tool, youtube_tool]
executor_llm = get_llm().bind_tools(tools)

SYSTEM = """You are a task executor.

Your ONLY job is to call tools in the exact sequence given.

You MUST NOT summarize, analyze, answer, or reason.

Tool sequence: {tool_sequence}
YouTube URL: {yt_url}
File registry: {file_registry}
"""


def _get_filename_from_path(
    file_registry: dict,
    file_path: str,
) -> str | None:

    for filename, meta in file_registry.items():
        if meta.get("path") == file_path:
            return filename

    return None


def _parse_tool_results(
    state: State,
) -> dict:

    extracted_texts = dict(
        state.get("extracted_texts", {})
    )

    audio_transcript = state.get(
        "audio_transcript"
    )

    yt_transcript = state.get(
        "yt_transcript"
    )

    messages = state.get("messages", [])

    if not messages:
        return {}

    last_tool = messages[-1]

    if not isinstance(last_tool, ToolMessage):
        return {}

    try:
        data = json.loads(last_tool.content)
    except Exception:
        return {}

    # Find the tool call that generated this ToolMessage
    tool_call_id = last_tool.tool_call_id

    tool_path = None

    for msg in reversed(messages):

        if not isinstance(msg, AIMessage):
            continue

        for tc in getattr(msg, "tool_calls", []):

            if tc["id"] == tool_call_id:

                tool_path = tc["args"].get(
                    "file_path"
                )

                break

        if tool_path:
            break

    # extract_tool
    if (
        last_tool.name == "extract_tool"
        and "extracted_text" in data
    ):

        filename = _get_filename_from_path(
            state.get("file_registry", {}),
            tool_path,
        )

        if filename:

            extracted_texts[filename] = (
                data["extracted_text"]
            )

    # audio_tool
    elif (
        last_tool.name == "audio_tool"
        and "transcript" in data
    ):

        audio_transcript = data["transcript"]

    # youtube_tool
    elif (
        last_tool.name == "youtube_tool"
        and "transcript" in data
    ):

        yt_transcript = data["transcript"]

    return {
        "extracted_texts": extracted_texts,
        "audio_transcript": audio_transcript,
        "yt_transcript": yt_transcript,
    }


def executor_node(state: State) -> State:

    # Parse newly arrived tool results
    updates = _parse_tool_results(state)

    tool_sequence = state.get(
        "tool_sequence",
        []
    )

    if not tool_sequence:
        return {
            **state,
            **updates,
        }

    system = SYSTEM.format(
        tool_sequence=json.dumps(
            tool_sequence
        ),
        yt_url=state.get(
            "yt_url",
            ""
        ),
        file_registry=json.dumps(
            state.get(
                "file_registry",
                {}
            ),
            indent=2,
        ),
    )

    all_messages = (
        [SystemMessage(content=system)]
        + state["messages"]
    )

    response = executor_llm.invoke(
        all_messages
    )

    remaining = (
        tool_sequence[1:]
        if response.tool_calls
        else []
    )

    return {
        **state,
        **updates,
        "messages": [response],
        "tool_sequence": remaining,
    }
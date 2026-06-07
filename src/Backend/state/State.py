from typing_extensions import TypedDict, List
from typing import Annotated, Optional
from operator import add
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[List, add_messages]
    raw_text: str
    uploaded_files: dict
    file_registry: dict

    tool_call: str
    next_node: str
    planner_reasoning: str
    is_done: bool

    extracted_texts: dict
    ocr_confidences: dict
    audio_transcript: Optional[str]
    yt_transcript: Optional[str]

    needs_clarification: bool
    follow_up_question: Optional[str]

    errors: list
    last_node_output: str

    plan_trace: Annotated[list, add]
    final_response: str

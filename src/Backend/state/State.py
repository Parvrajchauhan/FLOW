from typing_extensions import TypedDict, List
from typing import Annotated
from langgraph.graph.message import add_messages

def append_list(left: list | None, right: list | None) -> list:
    if not left:
        left = []
    if not right:
        right = []
    return left + right

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
    audio_transcript: str
    yt_transcript: str

    needs_clarification: bool
    follow_up_question: str

    errors: list
    last_node_output: str

    plan_trace:Annotated[list[str], append_list]  
    final_response: dict
    
    invoke_summary: str

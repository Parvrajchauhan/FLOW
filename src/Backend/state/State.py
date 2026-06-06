from typing_extensions import TypedDict,List
from typing import Annotated
from operator import add
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[List, add_messages] 
    raw_text: str
    uploaded_files: dict[str, bytes]
    file_registry: dict[str, dict]
    
    tool_sequence: list[str]
    specialist: str
    planner_reasoning: str
    is_done: bool
    
    extracted_texts: dict[str, str]
    ocr_confidences: dict[str, float | None]
    audio_transcript: str | None
    yt_transcript: str | None
    
    needs_clarification: bool
    follow_up_question: str | None
 
    errors: list[str]
    
    plan_trace:Annotated[list[str], add]
    next_node: str
    final_response: str
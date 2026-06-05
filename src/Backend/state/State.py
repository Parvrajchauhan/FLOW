from typing_extensions import TypedDict,List
from typing import Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List, add_messages] 
    raw_text: str
    extracted_texts: dict[str, str]
    audio_transcript: str | None
    audio_duration_s: float | None
    tool_sequence: list[str]
    planner_reasoning: str
    needs_clarification: bool
    follow_up_question: str | None
    ocr_confidence: float | None  
    errors: list[str]
    plan_trace: list[str]
    final_response: str
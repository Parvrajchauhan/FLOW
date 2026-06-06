from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from src.Backend.LLMs.geminiLLM import get_llm
from src.Backend.state.State import State
from typing import TypedDict, List, Optional
import json

llm = get_llm()

SYSTEM = """
You are a task planner for a multi-modal AI agent. You will receive the user's query and a dictionary of uploaded files containg base64-encoded data and type.

Available tools that excutor node can call:
- extract_tool   : extracts text from image or PDF files
- audio_tool     : transcribes audio files to text
- youtube_tool   : fetches transcript from a YouTube URL

Available specialist nodes  that excutor node  can route to after tool execution:
- summarize_node   : produces 1-line summary + 3 bullets + 5-sentence summary
- code_node        : Explain what code does, detect bugs, and mention time complexity. 
- sentiment_node   : performs sentiment analysis and give Label + confidence + one-line justification.
- cross_input_node : reasons across multiple sources that user upload and stored in the file registry(can be of diffent types so can have need to call diffrent tools) to answer a unified query.
- formatter_node     : if no specialist node is needed, route here to format the final answer for the user. for normal Q/A query.

Your job is to create a plan for how to execute the user's request. Analyze the user's query and the uploaded files, then decide:
1. Decide which tools need to run first (to extract content).
2. Decide which specialist node should handle the final reasoning.
3. If the user's intent is unclear or multiple tasks are equally plausible, set needs_clarification to true.

Return ONLY valid structured output
example output:
{{
  "tool_sequence": ["extract_tool", "youtube_tool"],
  "specialist": "summarize_node",
  "needs_clarification": false,
  "follow_up_question": null,
  "reasoning": "PDF contains a YouTube URL, user wants a summary of the video"
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  (only include if a YouTube URL is detected in the raw_text or extracted text)
}}

Rules:
- tool_sequence can be empty [] if no files uploaded and no URL detected
- specialist must be exactly one of the four node names above
- if needs_clarification is true, set follow_up_question to a short clear question and set tool_sequence to [] and specialist to null
- detect YouTube URLs in raw_text — if found put it in url
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "User query: {raw_text}\n\nFile registry: {file_registry}"),
])

class PlannerOutput(TypedDict):
    tool_sequence: List[str]
    specialist: str
    needs_clarification: bool
    follow_up_question: Optional[str]
    reasoning: str
    url: Optional[str]
    
    
planner_llm = llm.with_structured_output(PlannerOutput, include_raw=False)


def planner_node(state: State) -> State:
    chain = PROMPT | planner_llm

    try:
        plan = chain.invoke({"raw_text": state["raw_text"],
            "file_registry": json.dumps({k: v["type"] for k, v in state.get("file_registry", {}).items()}),})

    except Exception as e:
        plan = {
            "tool_sequence": [],
            "specialist": "code_node",
            "needs_clarification": False,
            "follow_up_question": None,
            "reasoning": f"Planner failed: {str(e)}",
            "url": None,
        }

    print("Planner Output:", plan)
    
    return {
        "messages": [AIMessage(content=str(plan))],
        "tool_sequence": plan.get("tool_sequence", []),
        "specialist": plan.get("specialist", "code_node"),
        "needs_clarification": plan.get("needs_clarification", False),
        "follow_up_question": plan.get("follow_up_question", None),
        "plan_trace": [f"Planner: {plan.get('reasoning', '')}"],
        "errors": [],
    }
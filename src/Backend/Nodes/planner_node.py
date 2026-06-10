from langchain_core.prompts import ChatPromptTemplate
from src.Backend.LLMs.geminiLLM import get_llm
from src.Backend.state.State import State
from typing import TypedDict, Optional
from langchain_core.messages import AIMessage, ToolMessage
import json

llm = get_llm()

SYSTEM = """You are a task planner for a multi modal AI agent. You will receive the user's query,
a dictionary of uploaded files containing base64 encoded data and type, audio_transcription and youtube_transcript.

Available tools that executor node can call:
(extract_tool - extracts text from image or PDF files)
(audio_tool - transcribes audio files to text)
(youtube_tool - fetches transcript from a YouTube URL)

Available specialist nodes:
(summarize_node - produces 1-line summary + 3 bullets + 5-sentence summary)
(code_node - Explain what code does, detect bugs, and mention time complexity.)
(sentiment_node - performs sentiment analysis and gives Label + confidence + one-line justification.)
(cross_input_node - reasons across multiple sources that user uploaded and stored in the file registry)
(formatter_node - if no specialist node is needed, route here to format the final answer for the user)
(qa_node - to answer general query of user)

Your job is to decide what to do next by analysing all the input.

IMPORTANT RULES:
- If files exist in file_registry but their content is NOT yet in extracted_texts, you MUST call extract_tool or audio_tool first.
- For audio files, call audio_tool first before any analysis.
- For PDF/image files, call extract_tool first before any analysis.
- Only call youtube_tool if there is a YouTube URL in the query or extracted texts AND the user wants it processed.
- After all content is extracted, route to the appropriate specialist node.
- If the user wants a summary → summarize_node.
- If the user asks about code → code_node.
- If the user wants sentiment → sentiment_node.
- If the user asks to compare/combine multiple inputs → cross_input_node.
- If the user has a general question → qa_node.
- When all tasks are complete → set is_done=True and next_node=formatter_node.
- In tool_call tell executor what tool to call 
- If there is no tool_call, set it to None and put next node in next_node.
- If needs_clarification is true: set follow_up_question and set next_node to null.

You will also receive results from previously completed steps in 'completed_steps'.
If all tasks are done and the user's request is fully satisfied, set is_done to true.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human","User query: {raw_text}\n\n"
     "File registry: {file_registry}\n\n"
     "Already extracted texts: {extracted_texts}\n\n"
     "Audio transcript: {audio_transcript}\n\n"
     "YouTube transcript: {yt_transcript}\n\n"
     "Completed steps so far: {plan_trace}\n\n"
     "Last node output: {last_node_output}")
])


class PlannerOutput(TypedDict):
    tool_call: Optional[str]
    next_node: Optional[str]
    needs_clarification: bool
    follow_up_question: Optional[str]
    reasoning: str
    url: Optional[str]
    is_done: bool


planner_llm = llm.with_structured_output(PlannerOutput, include_raw=False)


async def planner_node(state: State) -> dict:
    chain = PROMPT | planner_llm

    last_node_output = state.get("last_node_output", "none")

    if last_node_output == "none":
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, (AIMessage, ToolMessage)):
                last_node_output = str(msg.content)[:2000]
                break

    try:
        plan = await chain.ainvoke({
            "raw_text": state["raw_text"],
            "file_registry": json.dumps({k: v["type"] for k, v in state.get("file_registry", {}).items()}),
            "extracted_texts": json.dumps(state.get("extracted_texts", {})),
            "audio_transcript": state.get("audio_transcript") or "none",
            "yt_transcript": state.get("yt_transcript") or "none",
            "plan_trace": json.dumps(state.get("plan_trace", [])),
            "last_node_output": last_node_output,
        })
    except Exception as e:
        plan = {
            "tool_call": None,
            "next_node": "qa_node",
            "needs_clarification": False,
            "follow_up_question": None,
            "reasoning": f"Planner failed: {str(e)}",
            "url": None,
            "is_done": True,
        }
        
    planner_message = AIMessage(content=f"[PLANNER]\n{plan.get('reasoning', '')}")
    
    print("Planner next_node:", plan.get("next_node"))
    print("Planner tool_call:", plan.get("tool_call"))
    print("Planner reasoning:", plan.get("reasoning"))

    return {
        "messages": [planner_message],
        "tool_call": plan.get("tool_call") or "",
        "next_node": plan.get("next_node") or "qa_node",
        "needs_clarification": plan.get("needs_clarification", False),
        "follow_up_question": plan.get("follow_up_question"),
        "planner_reasoning": plan.get("reasoning", ""),
        "url": plan.get("url"),
        "plan_trace": [f"Planner: {plan.get('reasoning', '')}"],
        "is_done": plan.get("is_done", False),
        "errors": [],
    }

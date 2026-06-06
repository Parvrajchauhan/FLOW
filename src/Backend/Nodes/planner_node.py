from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from src.Backend.LLMs.geminiLLM import llm
from src.Backend.state.State import AgentState
import json
import re

PLANNER_SYSTEM = """
You are a task planner for a multi-modal AI agent.
You will receive the user's query and a registry of uploaded files.

Available tools (pure computation, no LLM):
- extract_tool   : extracts text from image or PDF files
- audio_tool     : transcribes audio files to text
- youtube_tool   : fetches transcript from a YouTube URL

Available specialist nodes (LLM reasoning):
- summarize_node   : produces 1-line summary + 3 bullets + 5-sentence summary
- qa_node          : answers questions grounded in extracted context
- analyze_node     : sentiment analysis OR code explanation (set mode accordingly)
- cross_input_node : reasons across multiple sources to answer a unified query

Your job:
1. Decide which tools need to run first (to extract content).
2. Decide which specialist node should handle the final reasoning.
3. If the user's intent is unclear or multiple tasks are equally plausible, set needs_clarification to true.

Return ONLY valid JSON. No preamble. No markdown. No explanation.

{
  "tool_sequence": ["extract_tool", "youtube_tool"],
  "specialist": "summarize_node",
  "needs_clarification": false,
  "follow_up_question": null,
  "reasoning": "PDF contains a YouTube URL, user wants a summary of the video"
}

Rules:
- tool_sequence can be empty [] if no files uploaded and no URL detected
- specialist must be exactly one of the four node names above
- if needs_clarification is true, set follow_up_question to a short clear question and set tool_sequence to [] and specialist to null
- detect YouTube URLs in raw_text — if found, always include youtube_tool in tool_sequence
"""

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM),
    ("human", "User query: {raw_text}\n\nFile registry: {file_registry}"),
])

planner_llm = llm  # no tools bound — pure reasoning only


def planner_node(state: AgentState) -> AgentState:
    chain = PLANNER_PROMPT | planner_llm

    response = chain.invoke({
        "raw_text": state["raw_text"],
        "file_registry": json.dumps({
            k: v["type"] for k, v in state.get("file_registry", {}).items()
        }),
    })

    raw = response.content.strip()

    # strip accidental markdown fences
    raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        # fallback — send to qa_node with no tools
        plan = {
            "tool_sequence": [],
            "specialist": "qa_node",
            "needs_clarification": False,
            "follow_up_question": None,
            "reasoning": "JSON parse failed, defaulting to QA",
        }

    return {
        **state,
        "messages": [AIMessage(content=raw)],
        "tool_sequence": plan.get("tool_sequence", []),
        "specialist": plan.get("specialist", "qa_node"),
        "needs_clarification": plan.get("needs_clarification", False),
        "follow_up_question": plan.get("follow_up_question"),
        "plan_trace": [f"Planner: {plan.get('reasoning', '')}"],
        "errors": [],
    }
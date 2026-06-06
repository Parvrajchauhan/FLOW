import json

from langchain_core.messages import ToolMessage
from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm

llm = get_llm()

SYSTEM = """
You are a summarizer. Return ONLY valid answers struture:
- one_liner 
- bullets 
- five_sentences 
 no more then this just: one line summary, Three bullets, Five sentence summary.
"""




def summarize_node(state: State) -> dict:

    content_to_summarize = ""

    for msg in state["messages"]:

        if not isinstance(msg, ToolMessage):
            continue

        try:
            data = json.loads(msg.content)

            if not isinstance(data, dict):
                continue

            content_to_summarize = (
                data.get("transcript")
                or data.get("extracted_text")
                or data.get("text")
                or data.get("content")
                or ""
            )

            if content_to_summarize:
                break

        except Exception:
            content_to_summarize = msg.content
            break

    if not content_to_summarize:
        return {
            "errors": [
                "No content found for summarization."
            ]
        }

    prompt = f"""
{SYSTEM}

Content:
{content_to_summarize}
"""

    response = llm.invoke(prompt)

    return {
        "messages": [response],
        "plan_trace": state.get("plan_trace", [])
        + ["Summarize Node: generated summary"]
    }
import json

from langchain_core.messages import ToolMessage
from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm

llm = get_llm()

SYSTEM = """
You are a code reviewer.

Return ONLY:
1. language
2. explanation
3. bugs
4. time_complexity
5. space_complexity
"""


def code_node(state: State) -> dict:
    """Review code extracted from text/OCR/PDF/tool outputs."""

    code_text = ""

    for msg in reversed(state["messages"]):

        if not isinstance(msg, ToolMessage):
            continue

        try:
            data = json.loads(msg.content)

            if isinstance(data, dict):
                code_text = (
                    data.get("extracted_text")
                    or data.get("transcript")
                    or data.get("content")
                    or ""
                )

                if code_text:
                    break

        except Exception:
            pass

    if not code_text:
        return {
            "errors": ["No code found to review."]
        }

    prompt = f"""
{SYSTEM}

Analyze this code:

{code_text}
"""

    response = llm.invoke(prompt)

    return {
        "messages": [response],
        "plan_trace": state.get("plan_trace", [])
        + ["Code Node: reviewed extracted code"]
    }
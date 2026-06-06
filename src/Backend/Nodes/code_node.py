import json

from langchain_core.messages import ToolMessage
from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm

llm = get_llm()

SYSTEM = """You are a code reviewer.
do code review and return ONLY:
language, explanation, bugs, time_complexity, space_complexity"""

def code_node(state: State) -> dict:
    extracted = state.get("extracted_texts", {})
    raw=state.get("raw_text", "")
    if not extracted:
        return {"errors": ["No code found to review."]}

    code_text = "\n\n".join( f"SOURCE: {name}\n{text}" for name, text in extracted.items())

    response = llm.invoke(f"{SYSTEM}\n\nAnalyze this code:\n\n{code_text} \n\n raw query: {raw}")


    return {"messages": [response],"plan_trace": ["Code_Node:Code is reviewed"],
        }
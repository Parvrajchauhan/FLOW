from langchain_core.messages import SystemMessage
from src.Backend.LLMs.geminiLLM import get_llm
from src.Backend.state.State import State

llm = get_llm()

SYSTEM = """You are a response formatter. You receive a final answer and must present it 
cleanly and clearly to the user. Do not add unnecessary content — just clean up formatting, 
fix any markdown issues, and ensure it reads well.

Output only the final, user-facing response. Do not add meta-commentary.
"""


def formatter_node(state: State) -> dict:
    final_response = state.get("final_response", "")
    last_node_output = state.get("last_node_output", "")

    content = final_response or last_node_output

    if not content:
        # Build from available data
        parts = []
        if state.get("extracted_texts"):
            parts.append("I extracted the content from your file(s).")
        content = " ".join(parts) if parts else "I have processed your request."

    # Light cleanup pass
    messages = [
        SystemMessage(content=SYSTEM),
        {"role": "user", "content": f"Format this response for the user:\n\n{content}"},
    ]

    response = llm.invoke(messages)
    result = response.content if hasattr(response, "content") else str(response)

    return {
        "final_response": result,
        "last_node_output": result,
        "plan_trace": ["Formatter_Node: Final response formatted and ready"],
    }

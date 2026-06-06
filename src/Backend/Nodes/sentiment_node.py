from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm

llm = get_llm()

SYSTEM = """ You are a sentiment analyzer. Return ONLY: label (positive, negative or neutral), confidence (0.0 to 1.0), justification (one sentence) No extra text."""

def sentiment_node(state: State) -> dict:
    extracted = state.get("extracted_texts", {})
    raw=state.get("raw_text", "")
    text = ("\n\n".join(f"SOURCE: {name}\n{content}" 
                        for name, content in extracted.items())
                        if extracted else raw)

    response = llm.invoke(f"{SYSTEM}\n\nAnalyze sentiment of:\n\n{text}\n\n raw query: {raw}")

    return {"messages": [response]}
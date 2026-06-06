from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm

llm = get_llm()

SYSTEM = """
You are a sentiment analyzer.
Return ONLY:
 label (positive, negative or neutral), confidence (0.0 to 1.0), justification (one sentence)
No extra text.
"""

def sentiment_node(state: State) -> dict:
    """Returns sentiment label, confidence and justification."""

    text = state["text"]
        
    prompt = (SYSTEM + "\n\nAnalyze sentiment of:\n\n" + text)

    response = llm.invoke(prompt)

    return {"messages": response}
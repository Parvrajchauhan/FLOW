from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm

llm = get_llm()

SYSTEM = """You are a summarizer. Return ONLY valid answer structure: one liner summary, three bullets and five sentences summary. No more than this."""

def summarize_node(state: State) -> dict:
    extracted = state.get("extracted_texts", {})
    raw=state.get("raw_text", "")

    content_to_summarize = ("\n\n".join( f"SOURCE: {name}\n{content}"
                                        for name, content in extracted.items())
                                        if extracted else raw)

    if not content_to_summarize:
        return {"errors": ["No content found for summarization."]}

    response = llm.invoke(f"{SYSTEM}\n\nContent:\n{content_to_summarize} \n\n raw query: {raw}" )

    return {"messages": [response]}
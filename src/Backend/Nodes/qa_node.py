from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm

llm = get_llm()

SYSTEM = """You a helpful assistent who answer according to user query"""

def qa_node(state: State) -> dict:
    extracted = state.get("extracted_texts", {})
    raw_text = state.get("raw_text","")
    audio_transcript = state.get("audio_transcript")
    yt_transcript = state.get("yt_transcript")

    sources = []

    for name, content in extracted.items():
        if content and content.strip():
            sources.append(f"SOURCE [{name}]:\n{content}")

    if audio_transcript and audio_transcript.strip():
        sources.append(f"SOURCE [audio]:\n{audio_transcript}")

    if yt_transcript and yt_transcript.strip():
        sources.append(f"SOURCE [youtube]:\n{yt_transcript}")

    source_text = "\n\n".join(sources)

    response = llm.invoke(
        f"{SYSTEM}\n\n"
        f"Sources:\n\n{source_text}\n\n" 
        f"User Question:\n{raw_text}")

    return {"messages": [response]}
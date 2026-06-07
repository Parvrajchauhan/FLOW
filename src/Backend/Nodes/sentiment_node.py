from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm

llm = get_llm()

SYSTEM = """ You are a sentiment analyzer. Return ONLY: label (positive, negative or neutral), confidence (0.0 to 1.0), justification (one sentence) No extra text."""

def sentiment_node(state: State) -> dict:
    extracted = state.get("extracted_texts", {})
    raw_text = state.get("raw_text","Analyze all available sources.")
    audio_transcript = state.get("audio_transcript","")
    yt_transcript = state.get("yt_transcript","")

    sources = []
    
    for name, content in extracted.items():
        if content and content.strip():
            sources.append(f"SOURCE [{name}]:\n{content}")

    if audio_transcript and audio_transcript.strip():
        sources.append(f"SOURCE [audio]:\n{audio_transcript}")

    if yt_transcript and yt_transcript.strip():
        sources.append(f"SOURCE [youtube]:\n{yt_transcript}")
        
    source_text = "\n\n".join(sources)
        
    response = llm.invoke(f"{SYSTEM}\n\nAnalyze sentiment of: \n{source_text} \n\n raw query: {raw_text}" )

    return {"messages": [response],"plan_trace":["Sentiment_node: sentiment is determined"],
        }
from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm

llm = get_llm()

SYSTEM = """You are a summarizer. Return ONLY valid answer structure: one liner summary, three bullets and five sentences summary. No more than this."""

def summarize_node(state: State) -> dict:
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
        
    response = llm.invoke(f"{SYSTEM}\n\nextracted content:\n{source_text} \n\n raw query: {raw_text}" )


    return {"messages": [response],"plan_trace":  ["Summarize_node: Text is Summarized"],
        }
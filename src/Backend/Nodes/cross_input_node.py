from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm
from langchain_core.messages import SystemMessage

llm = get_llm()

SYSTEM = """You are a multi-source analyst. You will receive information extracted from multiple sources
        (PDFs, images, audio transcripts, OCR outputs, etc.). Analyze information across ALL available sources based on user query.
        Return ONLY: analysis, sources_used, common_themes, differences
        Provide:
1. A clear comparative or combined analysis
2. Key similarities and differences (if applicable)
3. A unified conclusion answering the user's query
"""

def cross_input_node(state: State) -> dict:
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

    if not sources:
        return {"errors": ["No extracted content available for cross-source analysis."]}

    source_text = "\n\n".join(sources)

    messages = [
        SystemMessage(content=SYSTEM),
        {"role": "user", "content": f"User Query: {raw_text}\n\nSources:\n\n{source_text}"},
    ]
    response = llm.invoke(messages)

    return {"messages": [response],"plan_trace":  ["Cross_query_Node: analysis is done"],
        }
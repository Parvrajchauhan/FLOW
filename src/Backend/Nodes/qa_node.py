from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm
from langchain_core.messages import SystemMessage

llm = get_llm()

SYSTEM = """You are a helpful, friendly conversational AI assistant. 
Answer the user's question clearly and helpfully.
If there is extracted content available, use it to answer more accurately.
Keep responses concise but complete.
IMPORTANT:
Return response as plain text suitable for display inside a textarea.
No markdown formatting whatsoever."""

async def qa_node(state: State) -> dict:
    extracted = state.get("extracted_texts", {})
    raw_text = state.get("raw_text","")
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

    user_message = f"Query: {raw_text}"
    if source_text:
        user_message += f"\n\nContext:\n{source_text}"

    messages = [
        SystemMessage(content=SYSTEM),
        {"role": "user", "content": user_message},
    ]

    response = await llm.ainvoke(messages)
    return {"messages": [response],"plan_trace": ["QA_Node: query is answered"],
        }
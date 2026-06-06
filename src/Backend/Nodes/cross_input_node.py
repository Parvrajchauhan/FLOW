from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm

llm = get_llm()

SYSTEM = """
You are a multi-source analyst.

You will receive information extracted from multiple sources
(PDFs, images, audio transcripts, OCR outputs, etc.).

Analyze information across ALL available sources.

Return ONLY:

1. analysis
2. sources_used
3. common_themes
4. differences
"""


def cross_input_node(state: State) -> dict:
    """
    Analyze information across all extracted sources.
    """

    source_text = ""
    sources_used = []

    # PDF/Image/OCR outputs
    for source_name, content in state.get(
        "extracted_texts", {}
    ).items():

        if content and content.strip():

            source_text += (
                f"SOURCE [{source_name}]:\n"
                f"{content}\n\n"
            )

            sources_used.append(source_name)

    # Audio transcript
    if state.get("audio_transcript"):

        source_text += (
            "SOURCE [audio]:\n"
            f"{state['audio_transcript']}\n\n"
        )

        sources_used.append("audio")

    if not source_text.strip():
        return {
            "errors": [
                "No extracted content available for cross-source analysis."
            ]
        }

    question = state.get(
        "raw_text",
        "Analyze all available sources."
    )

    prompt = f"""
{SYSTEM}

Sources:

{source_text}

User Question:
{question}
"""

    response = llm.invoke(prompt)

    return {
        "messages": [response],
        "plan_trace": state.get("plan_trace", [])
        + [
            f"Cross Input Node: analyzed {len(sources_used)} sources"
        ]
    }
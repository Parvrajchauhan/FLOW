from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm

llm = get_llm()

SYSTEM = """
You are a multi-source analyst. You will be given content from multiple sources. Analyze information across ALL sources and return ONLY:
analysis, sources_used, common_themes, differences
"""


def cross_input_node(state: State) -> dict:
    """Analyze information from multiple sources and answer the user's question by reasoning across all sources."""

    sources = state["sources"]
    question = state["question"]

    source_text = ""

    for source_name, content in sources.items():
        if content and content.strip():
            source_text += (
                f"SOURCE [{source_name}]:\n"
                f"{content}\n\n"
            )

    prompt = (SYSTEM+ "\n\n"+ source_text+ f"Question:\n{question}")

    response = llm.invoke(prompt)

    return {"messages": response}
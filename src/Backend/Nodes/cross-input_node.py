from src.Backend.state.State import State


SYSTEM = """
You are a multi-source analyst.

You will be given content from multiple sources.

Analyze information across ALL available sources and return ONLY:

- analysis
- sources_used
- common_themes
- differences

No extra text.
"""


class CrossInputNode:
    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        """
        Analyze information from multiple sources and answer
        the user's question by reasoning across all sources.

        Input:
            sources = {
                "pdf": "...",
                "audio": "...",
                "youtube": "..."
            }

            question = "..."

        Output:
            analysis
            sources_used
            common_themes
            differences
        """

        sources = state["sources"]
        question = state["question"]

        source_text = ""

        for source_name, content in sources.items():
            if content and content.strip():
                source_text += (
                    f"SOURCE [{source_name}]:\n"
                    f"{content}\n\n"
                )

        prompt = (
            SYSTEM
            + "\n\n"
            + source_text
            + f"Question:\n{question}"
        )

        response = self.llm.invoke(prompt)

        return {"messages": response}
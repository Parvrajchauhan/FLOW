from src.Backend.state.State import State

SENTIMENT_SYSTEM = """
You are a sentiment analyzer.

Return ONLY:
- label (positive | negative | neutral)
- confidence (0.0 to 1.0)
- justification (one sentence)

No extra text.
"""

CODE_SYSTEM = """
You are a code reviewer.

Return ONLY:
- language
- explanation
- bugs
- time_complexity
- space_complexity

No extra text.
"""


class AnalyzeNode:
    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        """
        Analyze text based on mode.

        mode = "sentiment"
            Returns sentiment label, confidence and justification.

        mode = "code"
            Returns language, explanation, bugs,
            time complexity and space complexity.
        """

        text = state["text"]
        mode = state["mode"]

        if mode == "sentiment":
            prompt = (
                SENTIMENT_SYSTEM
                + "\n\nAnalyze sentiment of:\n\n"
                + text
            )

        elif mode == "code":
            prompt = (
                CODE_SYSTEM
                + "\n\nAnalyze this code:\n\n"
                + text
            )

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        response = self.llm.invoke(prompt)

        return {"messages": response}
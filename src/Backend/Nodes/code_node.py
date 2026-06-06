from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm

llm = get_llm()

SYSTEM = """You are a code reviewer. Return ONLY:
1. language
2. explanation
3. bugs
4. time_complexity
5. space_complexity
"""


def code_node(state: State) -> dict:
    """ Returns language, explanation, bugs,time complexity and space complexity."""

    text = state["text"]
       
    prompt = (SYSTEM+ "\n\nAnalyze this code:\n\n"+ text)

    response =llm.invoke(prompt)

    return {"messages": response}
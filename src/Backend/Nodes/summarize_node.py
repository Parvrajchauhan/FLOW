from src.Backend.state.State import State
from src.Backend.LLMs.geminiLLM import get_llm

llm = get_llm()

SYSTEM = """
You are a summarizer. Return ONLY valid answers struture:
- one_liner 
- bullets 
- five_sentences 
 no more then this just: one line summary, Three bullets, Five sentence summary.
"""

def summarize_node(state:State) ->dict:
    """
    This function processes the input state and generates a Summary using the LLM.
    it take the user input text amd out a summary in this format one line summary, Three bullets, Five sentence summary. 
    """
    conversation_history = state["messages"]
    prompt= SYSTEM + "\n\n" + "Here is the conversation history:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    response = llm.invoke(prompt)
        
    return {"messages": response}
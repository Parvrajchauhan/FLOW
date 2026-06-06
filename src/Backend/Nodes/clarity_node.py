from src.Backend.state.State import State
from langchain_core.messages import AIMessage

def clarify_node(state: State) -> dict:
    
    question = state["follow_up_question"]
    
    msg = AIMessage(content=question)
    
    return {
        "messages": [msg],
        "final_response": question,
    }
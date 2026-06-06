from src.Backend.state.State import State
from langchain_core.messages import HumanMessage
import base64

def ingest_node(state: State) -> State:
    combined = state["raw_text"]
    
    file_registry = {}
    
    for file_bytes in state["uploaded_files"].items():
        b64= base64.b64encode(file_bytes).decode() 
    
    messages = [HumanMessage(content=combined)]
    
    return {
        "messages": messages
    }
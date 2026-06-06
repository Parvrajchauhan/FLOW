from src.Backend.state.State import State
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def formatter_node(state: State) -> dict:
    plan_trace = []
    final_response = ""
    
    follow_up_question = state.get("follow_up_question")

    if follow_up_question:
        return {"plan_trace":["need clarification"],"final_response": final_response}
    

    return {"final_response": final_response}
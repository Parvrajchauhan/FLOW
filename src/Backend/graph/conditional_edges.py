from typing import Literal
from langchain_core.messages import ToolMessage, AIMessage
from src.Backend.state.State import State


def route_after_planner(state: State) -> Literal["clarify", "executor", "formatter"]:
    if state.get("needs_clarification", False):
        return "clarify"

    is_done = state.get("is_done", False)

    if not is_done:
        return "executor"
    return "formatter"


def route_after_executor(state: State) -> str:
    messages = state.get("messages", [])
    
    if messages and isinstance(messages[-1], AIMessage):
        tool_calls = getattr(messages[-1], "tool_calls", [])
        if tool_calls:
            return "tools"

    tool_call = str(state.get("tool_call", "")).strip().lower()

    if tool_call not in ["", "none", "null"]:
        return "planner"

    next_node = state.get("next_node", "qa_node")

    node_map = {
        "summarize_node": "summarize_node",
        "code_node": "code_node",
        "sentiment_node": "sentiment_node",
        "cross_input_node": "cross_input_node",
        "qa_node": "qa_node",
        "formatter_node": "formatter",
    }

    return node_map.get(next_node, "qa_node")

from typing import Literal
from src.Backend.state.State import State

def route_after_planner(state: State) -> Literal["clarify", "executor"]:
    """Planner decides whether more information is needed."""
    
    if state.get("needs_clarification", False): return "clarify"

    return "executor"


def route_after_executor(state: State) -> Literal["tools","summarize_node","cross_input_node","code_node","sentiment_node","formatter_node"]:
    """Executor decides the next node based on the generated plan."""
    
    tool_sequence = state.get("tool_sequence", [])

    if not tool_sequence:
        return "formatter_node"

    next_step = tool_sequence[0]

    routing_map = {"tools": "tools", "summarize": "summarize_node", "cross_input": "cross_input_node", "code": "code_node",
        "sentiment": "sentiment_node","formatter": "formatter_node"}

    return routing_map.get(next_step, "formatter_node")
from typing import Literal
from langchain_core.messages import ToolMessage
from src.Backend.state.State import State


def route_after_planner(state: State) -> Literal["clarify", "executor"]:
    if state.get("needs_clarification", False):
        return "clarify"
    
    return "executor"


def route_after_executor(state: State):
    tool_sequence = state.get("tool_sequence", [])
    messages = state.get("messages", [])

    if (messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls):
        return "tools"

    if not tool_sequence:
        return state.get("specialist")

    return "executor"
from src.Backend.state.State import State
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def formatter_node(state: State) -> dict:
    plan_trace = []
    final_response = ""

    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            plan_trace.append("Ingest: user input received")

        elif isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                plan_trace.append(f"Executor: called {tc['name']} "
                                  f"with args {list(tc['args'].keys())}")

        elif isinstance(msg, ToolMessage):
            plan_trace.append(f"Tool result: {msg.name} → success")

        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            final_response = msg.content
            plan_trace.append(f"Specialist: generated final response")

    return {"plan_trace": plan_trace,"final_response": final_response}
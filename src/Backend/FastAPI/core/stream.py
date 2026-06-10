import json
from langchain_core.messages import HumanMessage

from src.Backend.graph.graph import compiled_graph


def sse(event: str, data: dict) -> str:
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


async def stream_graph(user_message: str, session_id: str, uploaded_files: dict[str, bytes] | None = None):
    yield sse("ping", {"status": "stream_started"})

    config = ({"configurable": {"thread_id": session_id}} if session_id else {})

    initial_state = { "messages": [HumanMessage(content=user_message)], "uploaded_files": uploaded_files or {}}
    sent_traces = 0

    async for event in compiled_graph().astream_events(initial_state,config=config,version="v2"):
        kind = event["event"]
        name = event.get("name", "")

        if kind == "on_chain_end":
            output = event.get("data", {}).get("output")

            if isinstance(output, dict):
                traces = output.get("plan_trace", [])

                if (isinstance(traces, list) and name != "formatter" and len(traces) > sent_traces ):
                    new_traces = traces[sent_traces:]
                    sent_traces = len(traces)

                    yield sse("plan_trace",{"node": name,"steps": new_traces})

                if name == "formatter":
                    formatted = output.get("final_response")

                    if formatted:
                        yield sse("final_output", formatted)

        elif kind == "on_tool_start":
            yield sse("tool_start",{"tool": event.get("name"),
                                    "args": list(event.get("data", {}).get("input", {}).keys())})

        elif kind == "on_tool_end":
            yield sse("tool_end",{"tool": event.get("name"),
                                  "status": "success"})

    yield sse("done", {})
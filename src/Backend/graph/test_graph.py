from pathlib import Path
from pprint import pprint

from src.Backend.graph.graph import get_grap
from IPython.display import Image, display

graph = get_grap()


initial_state = {
    "raw_text": "who is india's current PM?",

    "uploaded_files": {},

    "messages": [],
    "file_registry": {},
    "tool_call": "",
    "next_node": "",
    "planner_reasoning": "",
    "is_done": False,

    "needs_clarification": False,
    "follow_up_question": None,

    "plan_trace": [],
    "errors": [],
    "url": None,

    # outputs
    "extracted_texts": {},
    "ocr_confidences": {},
    "audio_transcript": "",
    "yt_transcript": "",
    "final_response": "",
}

png_bytes = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_bytes)
    
for event in graph.stream(initial_state):
    print("\n" + "=" * 80)

    for node_name, node_output in event.items():
        print(f"NODE: {node_name}")

        if node_output is None:
            print(None)
            continue

        if "uploaded_files" in node_output:
            node_output = {
                **node_output,
                "uploaded_files": {
                    k: f"<{len(v)} bytes>"
                    for k, v in node_output["uploaded_files"].items()
                }
            }

        pprint(node_output)
from pathlib import Path
from pprint import pprint

from src.Backend.graph.graph import get_grap

graph = get_grap()

pdf_path = Path(
    r"C:\Users\parvr\Desktop\projects\FLOW\src\Backend\graph\sample.png"
)

pdf_bytes = pdf_path.read_bytes()

initial_state = {
    "raw_text": "do code review of code present in the image",

    "uploaded_files": {
        "sample.png": pdf_bytes
    },

    "messages": [],
    "file_registry": {},
    "tool_sequence": [],
    "specialist": "",
    "needs_clarification": False,
    "follow_up_question": None,
    "plan_trace": [],
    "errors": [],
    "url": None,

    # outputs
    "extracted_texts": {},
    "yt_transcript": "",
    "audio_transcript": "",
}
    
for event in graph.stream(initial_state):
    print("\n" + "=" * 80)

    for node_name, node_output in event.items():
        print(f"NODE: {node_name}")

        if node_output is None:
            print("WARNING: Node returned None")
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
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage

from src.Backend.state.State import State

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


async def ingest_node(state: State) -> State:
    combined = state.get("raw_text")

    if not combined:
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content:
                combined = msg.content
                break

    combined = combined or ""

    file_registry = {}

    for filename, file_bytes in state.get("uploaded_files", {}).items():

        ext = filename.rsplit(".", 1)[-1].lower()

        type_map = {"jpg": "image","jpeg": "image","png": "image","pdf": "pdf","mp3": "audio","wav": "audio","m4a": "audio"}

        saved_name = f"{uuid4()}.{ext}"
        saved_path = UPLOAD_DIR / saved_name

        with open(saved_path, "wb") as f:
            f.write(file_bytes)

        file_registry[filename] = {"path": str(saved_path),
                                   "type": type_map.get(ext, "unknown")}

    return {
        "raw_text": combined,
        "messages": [HumanMessage(content=combined)],
        "file_registry": file_registry,
        "uploaded_files": {},
    }
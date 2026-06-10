from typing import List

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from src.Backend.FastAPI.core.stream import stream_graph

router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post("/stream")
async def stream_agent(message: str = Form(...),session_id: str = Form("default"),files: List[UploadFile] | None = File(default=None)):
    uploaded_files: dict[str, bytes] = {}

    if files:
        for file in files:
            if file.filename:
                uploaded_files[file.filename] = await file.read()

    return StreamingResponse(
        stream_graph(user_message=message, session_id=session_id, uploaded_files=uploaded_files),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
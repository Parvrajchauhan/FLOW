from faster_whisper import WhisperModel
import tempfile
import os
import base64
from langchain_core.tools import tool

model = WhisperModel("tiny",device="cpu", compute_type="int8")

@tool
def audio_tool(b64: str) -> dict:
    """ Transcribe audio file (MP3/WAV/M4A) to text using Whisper. Call this when the user uploads an audio file.
    Returns transcript and duration in seconds. Always call this before summarize_node when audio is provided."""
    
    audio = base64.b64decode(b64)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(audio)
        tmp_path = f.name

    try:
        segments, info = model.transcribe(tmp_path)

        transcript = " ".join(segment.text for segment in segments)

        return {"transcript": transcript.strip()}

    finally:
        os.unlink(tmp_path)
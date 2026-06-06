from langchain_core.tools import tool
from faster_whisper import WhisperModel

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

@tool
def audio_tool(file_path: str) -> dict:
    """ Transcribe audio file (MP3/WAV/M4A) to text using Whisper. Call this when the user uploads an audio file.
    Returns transcript and duration in seconds. Always call this before summarize_node when audio is provided."""
    

    segments, info = model.transcribe(file_path)

    transcript = " ".join(segment.text.strip() for segment in segments)

    return {"transcript": transcript}
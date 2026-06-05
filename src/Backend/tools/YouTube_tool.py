import re
from typing import Dict, Any

from langchain_core.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url: str) -> str:
    patterns = [ r"(?:youtube\.com/watch\?v=)([\w-]+)", r"(?:youtu\.be/)([\w-]+)"]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL")


@tool
def youtube_tool(url: str) -> Dict[str, Any]:
    """
    Fetch transcript from a YouTube video form the  given URL. 
    
    Args: URL of youtube video
    
    Return: Transcript text and word count or error message if video not found or inavlid url.
    """

    try:
        video_id = extract_video_id(url)

        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id)

        transcript = " ".join( snippet.text for snippet in transcript_data )
        return { "transcript": transcript, "word_count": len(transcript.split()) }

    except Exception as e:
        return { "error": str(e) }
        
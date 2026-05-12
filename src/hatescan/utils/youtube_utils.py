import re

def extract_video_id(url: str) -> str:
    """
    Extracts the 11-character YouTube video ID from a given URL using Regex.
    Supports: youtube.com/watch?v=..., youtu.be/..., and youtube.com/embed/...
    """
    # Regex pattern to capture the ID after v=, /, embed/, or youtu.be/
    regex = r"(?:v=|\/|embed\/|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    
    match = re.search(regex, url)
    
    if match:
        return match.group(1)
    else:
        # Error handling if no valid ID is found
        raise ValueError("Could not extract a valid video ID from the provided URL.")
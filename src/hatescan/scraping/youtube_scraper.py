import os
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from datetime import datetime

# Import the utility to extract video IDs from URLs
from hatescan.utils.youtube_utils import extract_video_id

# Load environment variables from .env file
load_dotenv()

# Global configuration for quota safety
MAX_COMMENTS_PER_SEARCH = 20

def get_youtube_client():
    """
    Initializes and returns the YouTube Data API v3 client.
    Requires YOUTUBE_API_KEY to be set in the environment.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("Error: YOUTUBE_API_KEY not found in environment variables.")
        return None
    return build("youtube", "v3", developerKey=api_key)

def fetch_comments(video_input, max_results=MAX_COMMENTS_PER_SEARCH):
    """
    Fetches comments from a YouTube video handling both full URLs and raw IDs.
    
    Args:
        video_input (str): YouTube URL or 11-character video ID.
        max_results (int): Maximum number of comments to retrieve.
        
    Returns:
        list: Collection of dictionaries with comment metadata.
    """
    youtube = get_youtube_client()
    if not youtube:
        return []

    # Extract ID if a URL is provided, otherwise assume it's a raw ID
    try:
        if "youtube.com" in video_input or "youtu.be" in video_input:
            video_id = extract_video_id(video_input)
        else:
            video_id = video_input
    except ValueError as e:
        print(f"Input validation error: {e}")
        return []

    try:
        # Execute the API request to get top-level comments
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText"
        )
        response = request.execute()

        comments_data = []
        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            # Store structured data for the NLP pipeline
            comments_data.append({
                "video_id": video_id,
                "comment": snippet["textDisplay"],
                "author": snippet["authorDisplayName"],
                "published_at": snippet["publishedAt"]
            })
        
        return comments_data

    except HttpError as e:
        # Handle specific API errors (e.g., disabled comments, quota limit)
        print(f"YouTube API error: {e.resp.status} - {e.content}")
        return []
    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")
        return []

def save_to_csv(data, filename="youtube_comments.csv"):
    """
    Saves the list of comment dictionaries to a CSV file in data/raw/.
    """
    target_dir = os.path.join("data", "raw")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    df = pd.DataFrame(data)
    file_path = os.path.join(target_dir, filename)
    
    # Using utf-8 encoding to support emojis and special characters
    df.to_csv(file_path, index=False, encoding="utf-8")
    print(f"Successfully saved {len(data)} comments to: {file_path}")

if __name__ == "__main__":
    # Integration test with a full URL
    TEST_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 
    print(f"Running integration test with URL: {TEST_URL}")
    
    # Using the default MAX_COMMENTS_PER_SEARCH (20)
    comments = fetch_comments(TEST_URL)
    
    if comments:
        # Generate a unique filename using a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_to_csv(comments, filename=f"test_robustness_{timestamp}.csv")
    else:
        print("No comments extracted. Check the console for errors.")
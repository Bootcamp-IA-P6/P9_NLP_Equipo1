import os
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

def get_youtube_client():
    """Initializes the YouTube API client."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("Error: YOUTUBE_API_KEY not found.")
        return None
    return build("youtube", "v3", developerKey=api_key)

def fetch_comments(video_id, max_results=50):
    """Fetch comments from YouTube API."""
    youtube = get_youtube_client()
    if not youtube:
        return []

    try:
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
            # We collect text and some metadata for a better CSV
            comments_data.append({
                "video_id": video_id,
                "comment": snippet["textDisplay"],
                "author": snippet["authorDisplayName"],
                "published_at": snippet["publishedAt"]
            })
        
        return comments_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def save_to_csv(data, filename="youtube_comments.csv"):
    """Saves the extracted list of dictionaries to a CSV file in data/raw/."""
    # 1. Define the path (data/raw/)
    target_dir = os.path.join("data", "raw")
    
    # 2. Create the directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    # 3. Create a DataFrame and save to CSV
    df = pd.DataFrame(data)
    file_path = os.path.join(target_dir, filename)
    
    df.to_csv(file_path, index=False, encoding="utf-8")
    print(f"Successfully saved {len(data)} comments to: {file_path}")

if __name__ == "__main__":
    # Test with a specific video
    VIDEO_ID = "dQw4w9WgXcQ" 
    print(f"Starting extraction for video: {VIDEO_ID}...")
    
    # EXTRACT
    raw_data = fetch_comments(VIDEO_ID, max_results=20)
    
    # LOAD
    if raw_data:
        save_to_csv(raw_data, f"comments_{VIDEO_ID}_{datetime.now().strftime('%Y%m%d')}.csv")
    else:
        print("No data to save.")
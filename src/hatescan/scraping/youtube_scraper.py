import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
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
    # Import the utility to extract the ID from full URLs
    from hatescan.utils.youtube_utils import extract_video_id

    # List of YouTube video URLs or IDs to scrape
    urls_to_scrape = [
        "https://www.youtube.com/watch?v=yZ2p39BVwsE",        # Deportes (El Chiringuito)
        "https://www.youtube.com/watch?v=MY4ZUk9RVYg"         # Gaming (UrbVic)
    ]
    
    # Define the number of comments to extract per video
    MAX_COMMENTS_PER_VIDEO = 20

    print("Starting the multi-video scraping process...")

    # Iterate over each URL in the list
    for url in urls_to_scrape:
        print(f"\nProcessing target: {url}")
        
        # Clean the URL to obtain the 11-character video ID
        video_id = extract_video_id(url)
        
        if not video_id:
            print(f"Skipping: Could not extract a valid video ID from {url}")
            continue
            
        print(f"Extracted Video ID: {video_id}")
        
        # Fetch the comments using the cleaned ID
        print(f"Fetching up to {MAX_COMMENTS_PER_VIDEO} comments...")
        comments = fetch_comments(video_id, max_results=MAX_COMMENTS_PER_VIDEO)
        
        # If comments were found, save them to a unique CSV file
        if comments:
            output_filename = f"youtube_comments_{video_id}.csv"
            save_to_csv(comments, filename=output_filename)
        else:
            print(f"No comments extracted for video {video_id}")

    print("\nScraping process finished successfully.")
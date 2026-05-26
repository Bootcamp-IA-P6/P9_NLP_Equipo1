import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Secure the absolute path to the .env file located at the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.abspath(os.path.join(current_dir, "../../..", ".env"))

# Load environment variables explicitly from that path
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Critical Error: Missing Supabase credentials in .env file")

# Initialize the official Supabase client
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_hatescan_results(model_output_json: dict, current_session_user: str, video_title: str, video_id: str) -> bool:
    """
    Parses the transformer model output JSON and directly inserts the data 
    into the 'searches' and 'comments' relational tables in Supabase.
    
    Args:
        model_output_json (dict): The JSON dictionary received from the model inference.
        current_session_user (str): The identifier of the user currently logged into the app.
        video_title (str): The title of the analyzed YouTube video.
        video_id (str): The extracted unique ID of the YouTube video.
        
    Returns:
        bool: True if the database transaction succeeds, False otherwise.
    """
    try:
        # 1. Prepare data structure for the 'searches' table (The Header)
        search_row = {
            "user_id": None,  # Reserved for future features regarding search-specific user tracking
            "user_session": current_session_user,  # Identifier for the active application session
            "video_url": model_output_json.get("video_url"),
            "video_id": video_id,
            "video_title": video_title,
            "num_comments": model_output_json.get("total_comments"),  # Maps total_comments from JSON
            "model_used": model_output_json.get("model_used")         # Maps model_used from JSON
        }
        
        # Insert header into the searches table
        search_response = supabase_client.table("searches").insert(search_row).execute()
        
        if not search_response.data:
            print("Database Error: Failed to create the main record in 'searches'.")
            return False
            
        # Retrieve the auto-generated UUID search_id from the database response
        generated_search_id = search_response.data[0]["search_id"]
        print(f"Search record created successfully. ID: {generated_search_id}")
        
        # 2. Prepare bulk rows for the 'comments' table (The Details)
        comments_rows = []
        for item in model_output_json.get("comments", []):
            
            # Extraemos el diccionario de categorías (si no existe, usamos uno vacío)
            categories = item.get("categories", {})
            
            comment_row = {
                "comment_id": item.get("comment_id"),
                "search_id": generated_search_id,  
                "text_original": item.get("text_original"),
                "text_processed": None,  
                "is_toxic": item.get("is_toxic"),
                "confidence": float(item.get("confidence", 0.0)),  
                
                # Leemos desde el diccionario 'categories' hacia las columnas de Supabase
                "is_hatespeech": categories.get("is_hatespeech"),  
                "is_racist": categories.get("is_racist"),          
                "is_threat": categories.get("is_threat"),          
                "is_obscene": categories.get("is_obscene")         
            }
            comments_rows.append(comment_row)
            
        # Execute a bulk insert to optimize performance and reduce database round-trips
        if comments_rows:
            supabase_client.table("comments").insert(comments_rows).execute()
            print(f"Success: Inserted {len(comments_rows)} comment records linked to search ID.")
            
        return True

    except Exception as e:
        print(f"An unexpected error occurred while writing to Supabase: {e}")
        return False
import pandas as pd
import sys
import os

def check_setup():
    """
    Validates the Python environment and checks if the dataset is accessible.
    """
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    
    file_path = "youtoxic_english_1000.csv"
    
    # Check if the dataset exists in the root directory
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"Success: Dataset loaded correctly. Total rows: {len(df)}")
        except Exception as e:
            print(f"Error: Could not read the CSV file. {e}")
    else:
        print(f"Warning: {file_path} not found in the current directory.")

if __name__ == "__main__":
    check_setup()
import os
import urllib.request
import zipfile

def download_movielens_1m(data_dir="data"):
    """
    Downloads and extracts the MovieLens 1M dataset to the specified directory.
    
    Source: https://grouplens.org/datasets/movielens/1m/
    """
    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # URL for the MovieLens 1M dataset
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(data_dir, "ml-1m.zip")
    extract_dir = os.path.join(data_dir, "ml-1m")
    
    # Check if data already exists to avoid redundant downloads
    if os.path.exists(extract_dir):
        print(f"Dataset already exists at {extract_dir}. Skipping download.")
        return extract_dir
        
    print(f"Downloading MovieLens 1M from {url}...")
    try:
        # Download the zip file
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete. Extracting files...")
        
        # Extract the contents
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        print(f"Extraction complete. Data is ready in {extract_dir}.")
        
        # Clean up the zip file to save space
        os.remove(zip_path)
        print("Cleaned up the zip file.")
        
    except Exception as e:
        print(f"An error occurred during download/extraction: {e}")
        return None
        
    return extract_dir

def main(data_dir: str = "data") -> str | None:
    """Download and extract MovieLens 1M if not already present."""
    return download_movielens_1m(data_dir=data_dir)


if __name__ == "__main__":
    main()

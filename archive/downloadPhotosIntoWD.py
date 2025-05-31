import re
import zipfile
import os
import requests

def extract_file_id(url):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    match = re.search(r'id = ([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract file ID from the URL.")

def download_file_from_google_drive(file_id, destination):
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(URL , stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.status_code}")
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def download_and_extract(url, destination):
    file_id = extract_file_id(url)
    zip_path = os.path.join(destination, f"{file_id}.zip")
    
    print(f"Downloading file from Google Drive: {url}")
    download_file_from_google_drive(file_id, zip_path)
    
    print(f"Unzipping file to {destination}")
    unzip_file(zip_path, destination)
    
    os.remove(zip_path)  # Clean up the zip file after extraction
    print(f"Removed temporary zip file: {zip_path}")

if __name__ == "__main__": 
    url = "https://drive.google.com/drive/folders/1iTAez096wIsZuUvP7f8Hd-IfVm5mskBK?usp=drive_link"  # Replace with your Google Drive URL
    destination = "WorkDir"  # Replace with your desired destination directory
    
    os.makedirs(destination, exist_ok=True)
    
    download_and_extract(url, destination)
    print("Download and extraction completed successfully.")
    
    
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

def authenticate_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)

def download_images(folder_id, destination="WorkDir"):
    drive = authenticate_drive()
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    os.makedirs(destination, exist_ok=True)
    for file in file_list:
        if file['mimeType'].startswith('image/'):
            print(f"Downloading {file['title']}...")
            file.GetContentFile(os.path.join(destination, file['title']))

folder_id = "1iTAez096wIsZuUvP7f8Hd-IfVm5mskBK"
if __name__ == "__main__":
    download_images(folder_id)
    print("Download completed successfully.")
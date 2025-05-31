# FaceSearch

A deep learning-based face search application using two different models: **MobileFaceNet** and **InsightFace**.

## Features

- Download images from Google Drive using PyDrive.
- Detect and extract faces using MTCNN or InsightFace's built-in detector.
- Generate face embeddings with either MobileFaceNet or InsightFace.
- Search for similar faces in a directory using cosine similarity.
- Save matched results with bounding boxes.

## Model Usage

- **MobileFaceNet**:  
  Used in `searchUsingMobileNet.py`. This script uses MTCNN for face detection and MobileFaceNet for generating face embeddings. It is suitable for lightweight, efficient face recognition tasks.

- **InsightFace**:  
  Used in `searchUsingInsightFace.py`. This script leverages the InsightFace library for both face detection and embedding extraction, providing robust and accurate face recognition.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- opencv-python
- mtcnn
- insightface
- PyDrive

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/FaceSearch.git
   cd FaceSearch
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Google Drive API setup:**
   - Place your `client_secrets.json` in the project directory.
   - Enable the Google Drive API for your Google Cloud project.

4. **Download images from Google Drive:**
   ```bash
   python downloadPydrive.py
   ```

5. **Run face search with MobileFaceNet:**
   ```bash
   python searchUsingMobileNet.py
   ```

6. **Run face search with InsightFace:**
   ```bash
   python faceSearch.py
   ```

## File Structure

- `downloadPydrive.py` - Downloads images from a Google Drive folder.
- `searchUsingMobileNet.py` - Searches for similar faces using MobileFaceNet.
- `searchUsingInsightFace.py` - Searches for similar faces using InsightFace.
- `mobileFaceNet.py` - MobileFaceNet model definition.
- `client_secrets.json` - Google API credentials (not tracked by git).
- `output/` - Saved images with matched faces.

## Notes

- Make sure to update `reference_image_path` and `search_directory` in the scripts as needed.
- Model checkpoint file (`*.ckpt`) should be present in the project directory for MobileFaceNet.

## License

This project is for educational purposes.

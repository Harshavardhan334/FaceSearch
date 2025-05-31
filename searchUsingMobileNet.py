import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from mobileFaceNet import MobileFaceNet
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detector = MTCNN()

# Load model checkpoint
checkpoint = torch.load('068.ckpt', map_location=device)
model = MobileFaceNet()
model.load_state_dict(checkpoint['net_state_dict'])  # Adjust if your checkpoint has different keys
model.eval()
model.to(device)

# Transform: PIL image resize, ToTensor and normalization
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_face_embedding(image):
    # Convert BGR to RGB for MTCNN
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb_image)
    if not faces:
        return None

    # Use first detected face
    x1, y1, width, height = faces[0]['box']

    # Clip coordinates to image size to avoid errors
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x1 + width, image.shape[1])
    y2 = min(y1 + height, image.shape[0])

    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return None

    try:
        face_tensor = transform(face).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error transforming face: {e}")
        return None

    with torch.no_grad():
        embedding = model(face_tensor).cpu().numpy()
    return embedding

def search_faces(reference_image_path, search_directory, similarity_threshold=0.5):
    ref_image = cv2.imread(reference_image_path)
    if ref_image is None:
        print(f"Could not read reference image {reference_image_path}.")
        return

    ref_embedding = get_face_embedding(ref_image)
    if ref_embedding is None:
        print("No face found in the reference image.")
        return

    os.makedirs('output', exist_ok=True)

    for filename in os.listdir(search_directory):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(search_directory, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image {img_path}. Skipping.")
            continue

        face_embedding = get_face_embedding(img)
        if face_embedding is None:
            print(f"No face found in {filename}. Skipping.")
            continue

        similarity = cosine_similarity(ref_embedding, face_embedding)[0][0]

        if similarity >= similarity_threshold:
            print(f"Found similar face in {filename} with similarity {similarity:.2f}")

            # Draw bounding box on BGR image
            faces = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if faces:
                box = faces[0]['box']
                x1, y1 = max(0, box[0]), max(0, box[1])
                x2, y2 = x1 + box[2], y1 + box[3]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            output_path = os.path.join('output', f'matched_{filename}')
            cv2.imwrite(output_path, img)
            print(f"Saved matched face to {output_path}")
        else:
            print(f"No similar face found in {filename} (similarity: {similarity:.2f})")

    print("Search completed.")

if __name__ == "__main__":
    reference_image_path = 'IMG_0365.JPG'  # Path to the reference image
    search_directory = 'WorkDir'  # Directory containing images to search
    similarity_threshold = 0.5  # Similarity threshold for matching

    search_faces(reference_image_path, search_directory, similarity_threshold)

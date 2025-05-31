import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore", message=".*rcond.*")


reference_image_path =  'WIN_20250504_15_03_23_Pro.jpg' # Path to the reference image
search_directory = 'WorkDir'
similarity_threshold = 0.5  


app = FaceAnalysis( providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)


ref_img = cv2.imread(reference_image_path)
ref_faces = app.get(ref_img)

if not ref_faces:
    print("No face found in the reference image.")
    exit()

ref_face = ref_faces[0].embedding.reshape(1, -1)


for filename in os.listdir(search_directory):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(search_directory, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image {img_path}. Skipping.")
        continue

    faces = app.get(img)
    if not faces:
        print(f"No face found in {filename}. Skipping.")
        continue
    for face in faces:
        face_embedding = face.embedding.reshape(1, -1)
        similarity = cosine_similarity(ref_face, face_embedding)[0][0]

        if similarity >= similarity_threshold:
            print(f"Found similar face in {filename} with similarity {similarity:.2f}")
            box = face.bbox.astype(int)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            output_path = os.path.join('output', f'matched_{filename}')
            os.makedirs('Output', exist_ok=True)
            cv2.imwrite(output_path, img)
            print(f"Saved matched face to {output_path}")
        else:
            print(f"No similar face found in {filename} (similarity: {similarity:.2f})")
    print(f"Processed {filename}")

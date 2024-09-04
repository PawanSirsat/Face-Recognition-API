import os
import cv2
import numpy as np
from deepface import DeepFace

print("All libraries imported successfully!")

def load_images_from_folder(folder):
    """Load all images from a specified folder."""
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def encode_face(image, image_name):
    """Encode faces detected in an image using DeepFace."""
    try:
        # Detect faces in the image
        faces = DeepFace.extract_faces(img_path=image, detector_backend='opencv', enforce_detection=False)
        num_faces = len(faces)

        if num_faces == 0:
            print(f"No faces detected in {image_name}.")
            return None, 0
        else:
            print(f"Detected {num_faces} face(s) in {image_name}. Encoding the first detected face...")

            # Represent the first detected face in the image
            embedding = DeepFace.represent(image, model_name='Facenet', enforce_detection=False)[0]["embedding"]
            return np.array(embedding), num_faces

    except Exception as e:
        print(f"Error encoding face in {image_name}: {e}")
        return None, 0

def match_faces(user_encoding, folder_encodings, folder_filenames, tolerance=10.0):
    """Match the user face encoding with faces from the folder encodings."""
    matched_filenames = []
    for encoding, filename in zip(folder_encodings, folder_filenames):
        distance = np.linalg.norm(user_encoding - encoding)
        # print(f"Distance between user image and {filename}: {distance}")
        if distance < tolerance:
            matched_filenames.append(filename)
    return matched_filenames

def main():
    # Load the user image and encode it
    user_image_path = "./user/virat.png"  # Update to the correct path
    user_image = cv2.imread(user_image_path)
    if user_image is None:
        print(f"User image not found at {user_image_path}. Please check the path.")
        return

    user_encoding, num_faces = encode_face(user_image, user_image_path)
    if user_encoding is None or num_faces == 0:
        print("No face detected in the user image.")
        return
    elif num_faces > 1:
        print("Multiple faces detected in the user image. Please provide an image with a single face.")
        return
    else:
        print(f"Number of faces detected in user image: {num_faces}")

    # Load images from the collection folder and encode them
    folder_path = "./collection"  # Update to the correct path
    folder_images, folder_filenames = load_images_from_folder(folder_path)

    folder_encodings = []
    for image, filename in zip(folder_images, folder_filenames):
        encoding, num_faces = encode_face(image, filename)
        if encoding is not None:
            folder_encodings.append(encoding)
        else:
            print(f"No face detected in {filename}.")

    # Match user image encoding with folder encodings
    matched_filenames = match_faces(user_encoding, folder_encodings, folder_filenames)

    if matched_filenames:
        print("Matched images containing the same person as in user image:")
        for filename in matched_filenames:
            print(filename)
    else:
        print("No matching images found.")

if __name__ == "__main__":
    main()

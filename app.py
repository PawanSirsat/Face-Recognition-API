import logging
from flask import Flask, request, jsonify
import requests
import numpy as np
import cv2
from deepface import DeepFace

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

def download_image(url):
    try:
        response = requests.get(url)
        image_np = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        app.logger.error(f"Error downloading image from {url}: {e}")
        return None

def encode_face(image):
    try:
        faces = DeepFace.extract_faces(img_path=image, detector_backend='opencv', enforce_detection=False)
        num_faces = len(faces)
        if num_faces == 0:
            return None, 0
        else:
            embedding = DeepFace.represent(image, model_name='Facenet', enforce_detection=False)[0]["embedding"]
            return np.array(embedding), num_faces
    except Exception as e:
        app.logger.error(f"Error encoding face: {e}")
        return None, 0

def match_faces(user_encoding, folder_encodings, tolerance=10.0):
    matched_urls = []
    for encoding, url in folder_encodings:
        distance = np.linalg.norm(user_encoding - encoding)
        if distance < tolerance:
            matched_urls.append(url)
    return matched_urls

@app.route('/match_faces', methods=['POST'])
def match_faces_endpoint():
    data = request.json
    user_image_url = data.get('user_image_url')
    collection_image_urls = data.get('collection_image_urls', [])

    if not user_image_url or not collection_image_urls:
        app.logger.error('Missing user image URL or collection image URLs')
        return jsonify({'error': 'Missing user image URL or collection image URLs'}), 400

    user_image = download_image(user_image_url)
    if user_image is None:
        app.logger.error('User image could not be downloaded or processed')
        return jsonify({'error': 'User image could not be downloaded or processed'}), 400

    user_encoding, num_faces = encode_face(user_image)
    if user_encoding is None or num_faces != 1:
        app.logger.error('Face not detected in user image or multiple faces detected')
        return jsonify({'error': 'Face not detected in user image or multiple faces detected'}), 400

    folder_encodings = []
    for image_url in collection_image_urls:
        image = download_image(image_url)
        if image is not None:
            encoding, num_faces = encode_face(image)
            if encoding is not None:
                folder_encodings.append((encoding, image_url))
        else:
            app.logger.error(f"Image could not be downloaded or processed from {image_url}")

    matched_urls = match_faces(user_encoding, folder_encodings)

    return jsonify({'matched_image_urls': matched_urls})

if __name__ == "__main__":
    app.run(debug=True)

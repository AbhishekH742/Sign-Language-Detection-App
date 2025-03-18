from flask import Flask, request, jsonify
from flask_cors import CORS  # Enable CORS for frontend communication
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math
import base64
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load model and labels
model_path = os.path.join(os.getcwd(), "Model/keras_model.h5")
labels_path = os.path.join(os.getcwd(), "Model/labels.txt")

detector = HandDetector(maxHands=1)
classifier = Classifier(model_path, labels_path)

# Define labels
labels = ["Hello", "Please", "Thank You", "Victory", "Yes"]
imgSize = 300
offset = 20

def process_image(image):
    try:
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[max(0, y - offset): min(y + h + offset, img.shape[0]), 
                          max(0, x - offset): min(x + w + offset, img.shape[1])]
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            return labels[index]

        return None
    except Exception as e:
        return str(e)

@app.route('/')
def home():
    return "Sign Language Detection API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = base64.b64decode(data['image'])
        sign = process_image(image_data)

        if sign:
            return jsonify({'sign': sign})
        return jsonify({'error': 'No hand detected'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector 
from cvzone.ClassificationModule import Classifier
import math
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["Hello", "Please", "Thank You", "Victory", "Yes"]
imgSize = 300
offset = 20

def process_image(image):
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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = base64.b64decode(data['image'])
    sign = process_image(image_data)

    if sign:
        return jsonify({'sign': sign})
    return jsonify({'error': 'No hand detected'}), 400

if __name__ == '__main__':
    app.run(debug=True)

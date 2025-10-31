from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load YOLOv5 model
model = torch.hub.load('./yolov5', 'custom', path='model/yolov5s.pt', source='local')

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))

    results = model(image)
    detections = results.pandas().xyxy[0].to_dict(orient='records')

    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(debug=True)

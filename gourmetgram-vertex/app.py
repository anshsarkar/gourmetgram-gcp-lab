import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import base64
import io
import logging
from google.cloud import storage

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

GCS_STAGING_BUCKET = os.environ.get("GCS_STAGING_BUCKET", "")

CLASS_NAMES = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
    "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"]
CLASS_NAME_TO_DIR = {name: f"class_{i:02d}" for i, name in enumerate(CLASS_NAMES)}

def upload_to_gcs(file_bytes, predicted_class, filename):
    if not GCS_STAGING_BUCKET:
        return
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_STAGING_BUCKET)
        class_dir = CLASS_NAME_TO_DIR.get(predicted_class, "unknown")
        blob_name = f"incoming/{class_dir}/{uuid.uuid4().hex}_{filename}"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(file_bytes, content_type="image/jpeg")
    except Exception as e:
        logging.error(f"GCS upload failed: {e}")

model = models.mobilenet_v2(weights=None)
num_ftrs = model.last_channel
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_ftrs, 11)
)

# Vertex AI Endpoints mount model artifacts and set AIP_STORAGE_URI
# Fall back to bundled model for Cloud Run / GKE deployments
aip_storage = os.environ.get("AIP_STORAGE_URI", "")
if aip_storage:
    model_path = os.path.join(aip_storage, "food11.pth")
    logging.info(f"Loading model from Vertex AI artifact: {model_path}")
else:
    model_path = "food11.pth"
    logging.info("Loading bundled model: food11.pth")

state = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state)
model.eval()

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def model_predict(img_path, model):
    img = Image.open(img_path).convert('RGB')
    img = preprocess_image(img)

    classes = np.array(CLASS_NAMES)

    with torch.no_grad():
        output = model(img)
        prob, predicted_class = torch.max(output, 1)
    
    return classes[predicted_class.item()], torch.sigmoid(prob).item()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    preds = None
    if request.method == 'POST':
        f = request.files['file']
        save_path = os.path.join(app.instance_path, 'uploads', secure_filename(f.filename))
        f.save(save_path)
        preds, probs = model_predict(save_path, model)
        return '<button type="button" class="btn btn-info btn-sm">' + str(preds) + '</button>'
    return '<a href="#" class="badge badge-warning">Warning</a>'

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no JSON body"}), 400

    # Vertex AI Endpoints wraps requests: {"instances": [{"image": "..."}]}
    vertex_format = 'instances' in data
    if vertex_format:
        instance = data['instances'][0]
        if 'image' not in instance:
            return jsonify({"error": "missing 'image' field"}), 400
        img_bytes = base64.b64decode(instance['image'])
    elif 'image' in data:
        img_bytes = base64.b64decode(data['image'])
    else:
        return jsonify({"error": "missing 'image' field"}), 400

    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    temp_path = os.path.join(app.instance_path, 'uploads', f"{uuid.uuid4().hex}.jpg")
    img.save(temp_path)
    preds, probs = model_predict(temp_path, model)
    upload_to_gcs(img_bytes, preds, os.path.basename(temp_path))
    os.remove(temp_path)

    if vertex_format:
        return jsonify({"predictions": [{"prediction": preds, "confidence": probs}]})
    return jsonify({"prediction": preds, "confidence": probs})

@app.route('/event', methods=['POST'])
def handle_event():
    envelope = request.get_json()
    if not envelope or 'bucket' not in envelope or 'name' not in envelope:
        return jsonify({"error": "invalid event payload"}), 400

    source_bucket = envelope['bucket']
    object_name = envelope['name']

    if not object_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({"skipped": "not an image file"}), 200

    try:
        client = storage.Client()
        blob = client.bucket(source_bucket).blob(object_name)
        img_bytes = blob.download_as_bytes()

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        temp_path = os.path.join(app.instance_path, 'uploads', f"{uuid.uuid4().hex}.jpg")
        img.save(temp_path)
        preds, probs = model_predict(temp_path, model)
        upload_to_gcs(img_bytes, preds, os.path.basename(temp_path))
        os.remove(temp_path)

        logging.info(f"Eventarc: {source_bucket}/{object_name} -> {preds} ({probs:.2f})")
        return jsonify({"prediction": preds, "confidence": probs, "source": object_name})
    except Exception as e:
        logging.error(f"Eventarc processing failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    preds, probs = model_predict("./instance/uploads/test_image.jpeg", model)
    return str(preds)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)

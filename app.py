from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from phishing_detection import URLFeatureExtractor
import os
from werkzeug.utils import secure_filename
import cv2
from pyzbar.pyzbar import decode
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model and feature names
model_dir = 'models'
model_files = [f for f in os.listdir(model_dir) if f.startswith('phishing_detector_')]
feature_files = [f for f in os.listdir(model_dir) if f.startswith('feature_names_')]

# Get the latest model and feature names
latest_model = sorted(model_files)[-1]
latest_features = sorted(feature_files)[-1]

model = joblib.load(os.path.join(model_dir, latest_model))
feature_names = joblib.load(os.path.join(model_dir, latest_features))

# Initialize feature extractor
extractor = URLFeatureExtractor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def extract_url_from_qr(image_path):
    try:
        image = cv2.imread(image_path)
        decoded_objects = decode(image)
        if decoded_objects:
            return decoded_objects[0].data.decode('utf-8')
        return None
    except Exception as e:
        print(f"Error decoding QR code: {str(e)}")
        return None

def analyze_url(url):
    try:
        # Extract features
        features = extractor.extract_features(url)
        if features is None:
            return {"error": "Could not extract features from URL"}
            
        # Convert to DataFrame with correct feature order
        features_df = pd.DataFrame([features])[feature_names]
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        
        # Prepare feature information for display
        feature_info = {name: float(value) for name, value in features.items()}
        
        return {
            "is_phishing": bool(prediction),
            "confidence": float(probabilities[1]),
            "features": feature_info,
            "url": url
        }
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'url' in request.form:
        url = request.form['url']
        return jsonify(analyze_url(url))
    elif 'qr_image' in request.files:
        file = request.files['qr_image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract URL from QR code
            url = extract_url_from_qr(filepath)
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            if url:
                return jsonify(analyze_url(url))
            return jsonify({"error": "Could not extract URL from QR code"})
        return jsonify({"error": "Invalid file type"})
    
    return jsonify({"error": "No URL or QR code provided"})

if __name__ == '__main__':
    app.run(debug=True)

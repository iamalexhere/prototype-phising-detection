from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from phishing_detection import URLFeatureExtractor
import os
from werkzeug.utils import secure_filename
import cv2
from pyzbar.pyzbar import decode
import numpy as np
import logging
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Setup logging
def setup_app_logging(log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'webapp_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Web application logging setup complete. Log file: {log_file}")

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
setup_app_logging()

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
        logging.error(f"Error decoding QR code: {str(e)}")
        return None

def analyze_url(url):
    try:
        # Extract features
        features = extractor.extract_features(url)
        if features is None:
            return {
                'error': 'Could not extract features from URL'
            }
        
        # Create feature vector
        feature_vector = pd.DataFrame([features])
        
        # Remove non-numeric features
        if 'ip_address' in feature_vector.columns:
            feature_vector = feature_vector.drop('ip_address', axis=1)
            
        # Fill missing values
        feature_vector = feature_vector.fillna(-1)
        
        # Make prediction
        probability = model.predict_proba(feature_vector)[0][1]
        prediction = 1 if probability > 0.5 else 0
        
        # Get feature importance
        feature_importance = {
            name: float(value) for name, value in zip(feature_vector.columns, model.feature_importances_)
        }
        
        # Convert numpy values to Python native types
        def convert_to_native(value):
            if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                return int(value)
            if isinstance(value, (np.float64, np.float32)):
                return float(value)
            return value
        
        # Prepare DNS features for display
        dns_features = {
            'DNS Records': {
                'A Record': 'Present' if features.get('has_a_record', 0) == 1 else 'Missing',
                'MX Record': 'Present' if features.get('has_mx_record', 0) == 1 else 'Missing',
                'NS Record': 'Present' if features.get('has_ns_record', 0) == 1 else 'Missing'
            },
            'Domain Age': {
                'Age': f"{convert_to_native(features.get('domain_age_days', -1))} days",
                'Status': 'Young Domain' if features.get('is_domain_young', 1) == 1 else 'Established Domain'
            },
            'SSL Certificate': {
                'Status': 'Valid' if features.get('ssl_is_valid', 0) == 1 else 'Invalid',
                'Days Valid': convert_to_native(features.get('ssl_days_valid', -1)),
                'Self Signed': 'Yes' if features.get('ssl_is_self_signed', 1) == 1 else 'No'
            },
            'Registration': {
                'Has Registrar': 'Yes' if features.get('has_registrar', 0) == 1 else 'No',
                'Has Registrant': 'Yes' if features.get('has_registrant', 0) == 1 else 'No',
                'Days to Expiration': convert_to_native(features.get('days_to_expiration', -1))
            }
        }
        
        # Prepare all features for display
        all_features = {
            'URL Structure': {
                'URL Length': convert_to_native(features.get('url_length', 0)),
                'Domain Length': convert_to_native(features.get('domain_length', 0)),
                'Path Length': convert_to_native(features.get('path_length', 0)),
                'Subdomain Length': convert_to_native(features.get('subdomain_length', 0)),
                'TLD Length': convert_to_native(features.get('tld_length', 0)),
                'Domain Token Count': convert_to_native(features.get('domain_token_count', 0))
            },
            'Special Characters': {
                'Special Characters Count': convert_to_native(features.get('special_chars_count', 0)),
                'Digits Count': convert_to_native(features.get('digits_count', 0)),
                'Dots': convert_to_native(features.get('num_dots', 0)),
                'Hyphens': convert_to_native(features.get('num_hyphens', 0)),
                'Underscores': convert_to_native(features.get('num_underscores', 0)),
                'At Symbol (@)': 'Present' if features.get('has_at_symbol', 0) == 1 else 'Absent',
                'Percent (%)': convert_to_native(features.get('num_percent', 0)),
                'Ampersand (&)': convert_to_native(features.get('num_ampersand', 0)),
                'Hash (#)': convert_to_native(features.get('num_hash', 0))
            },
            'Security Indicators': {
                'HTTPS': 'Present' if features.get('has_https', 0) == 1 else 'Absent',
                'Is IP Address': 'Yes' if features.get('is_ip_address', 0) == 1 else 'No',
                'Is Private IP': 'Yes' if features.get('is_private_ip', 0) == 1 else 'No',
                'Query Components': convert_to_native(features.get('num_query_components', 0))
            }
        }
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'dns_features': dns_features,
            'feature_importance': feature_importance,
            'all_features': all_features
        }
        
    except Exception as e:
        logging.error(f"Error analyzing URL: {str(e)}")
        return {
            'error': str(e)
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'No URL provided'})
            
        logging.info(f"Analyzing URL: {url}")
        result = analyze_url(url)
        logging.info(f"Analysis result: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

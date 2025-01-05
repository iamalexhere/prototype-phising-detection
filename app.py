from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from phishing_detection import URLFeatureExtractor
import logging
from pathlib import Path
import json
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_latest_model(models_dir='models/split_10'):
    """Load the latest trained model and its feature names"""
    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory {models_dir} not found")
    
    model_files = list(models_path.glob("phishing_detector_split_9_*.joblib"))
    feature_files = list(models_path.glob("feature_names_split_9_*.joblib"))
    
    if not model_files or not feature_files:
        raise FileNotFoundError("Model or feature names file not found")
    
    latest_model = max(model_files, key=lambda x: x.name)
    latest_features = max(feature_files, key=lambda x: x.name)
    
    model = joblib.load(latest_model)
    feature_names = joblib.load(latest_features)
    
    return model, feature_names, latest_model.name

def normalize_feature_vector(feature_vector, feature_names):
    """Normalize features to a common scale"""
    normalized = feature_vector.copy()
    
    numeric_features = ['url_length', 'domain_length', 'path_length', 'query_length',
                       'fragment_length', 'domain_token_count', 'path_token_count',
                       'avg_domain_token_length', 'avg_path_token_length',
                       'domain_age_days', 'ssl_days_valid']
    
    for feature in numeric_features:
        if feature in feature_names:
            max_val = feature_vector[feature].max()
            if max_val > 0:
                normalized[feature] = feature_vector[feature] / max_val
    
    return normalized

def analyze_url(url):
    """Analyze URL and return phishing prediction results"""
    try:
        # Load model
        model, feature_names, model_name = load_latest_model()
        
        # Extract features
        extractor = URLFeatureExtractor()
        features = extractor.extract_features(url)
        
        if features is None:
            return {"error": "Failed to extract features from URL"}
            
        # Add is_private_ip feature
        if features.get('has_ip', 0):
            import re
            ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
            ip_match = re.search(ip_pattern, url)
            if ip_match:
                features['is_private_ip'] = 1 if is_private_ip(ip_match.group(0)) else 0
            else:
                features['is_private_ip'] = 0
        else:
            features['is_private_ip'] = 0
            
        # Prepare feature vector
        feature_dict = {name: 0 for name in feature_names}
        feature_dict.update({k: 1 if isinstance(v, bool) and v else 0 if isinstance(v, bool) else v 
                           for k, v in features.items()})
        feature_vector = pd.DataFrame([feature_dict])[feature_names]
        
        # Normalize features
        feature_vector = normalize_feature_vector(feature_vector, feature_names)
        
        # Get prediction probability
        probability = model.predict_proba(feature_vector)[0]
        phishing_prob = probability[1]
        
        # Calculate trust score
        trust_score = 0
        
        # SSL Group (~50% importance)
        if features.get('is_https', 0) and features.get('ssl_is_valid', 0):
            trust_score += 0.25
            normalized_ssl_days = feature_vector['ssl_days_valid'].iloc[0] if 'ssl_days_valid' in feature_names else 0
            if normalized_ssl_days > 0.5:
                trust_score += 0.15
        
        # URL Group (~29% importance)
        normalized_domain_length = feature_vector['domain_length'].iloc[0] if 'domain_length' in feature_names else 1
        if not features.get('has_multiple_subdomains', 0) and normalized_domain_length < 0.5:
            trust_score += 0.15
            
        # DNS Group (~11% importance)
        normalized_domain_age = feature_vector['domain_age_days'].iloc[0] if 'domain_age_days' in feature_names else 0
        if normalized_domain_age > 0.7:
            trust_score += 0.10
        if features.get('has_mx_record', 0) and features.get('has_ns_record', 0):
            trust_score += 0.10
            
        # Calculate adjusted probability
        adjusted_prob = max(0.01, min(0.99, phishing_prob - trust_score))
        is_phishing = 1 if adjusted_prob > 0.45 else 0
        
        # Prepare trust indicators
        trust_indicators = []
        if features.get('is_https', 0) and features.get('ssl_is_valid', 0):
            trust_indicators.append(f"Valid HTTPS/SSL certificate (valid for {features.get('ssl_days_valid', 0)} days)")
        if not features.get('has_multiple_subdomains', 0):
            trust_indicators.append("Simple domain structure (no multiple subdomains)")
        if features.get('domain_age_days', 0) > 365:
            trust_indicators.append(f"Domain age: {features['domain_age_days'] / 365:.1f} years")
        if features.get('has_mx_record', 0) and features.get('has_ns_record', 0):
            trust_indicators.append("Valid DNS records (MX and NS)")
            
        # Prepare warning indicators
        warning_indicators = []
        if features.get('has_ip', 0):
            warning_indicators.append("URL contains IP address")
        if features.get('has_at_symbol', 0):
            warning_indicators.append("URL contains @ symbol")
        if features.get('has_multiple_subdomains', 0):
            warning_indicators.append("Multiple subdomains detected")
        if features.get('is_domain_young', 0):
            warning_indicators.append("Domain is relatively new")
        if not features.get('ssl_is_valid', 0):
            warning_indicators.append("Invalid or missing SSL certificate")
            
        return {
            "url": url,
            "raw_probability": float(phishing_prob * 100),
            "trust_score": float(trust_score * 100),
            "adjusted_probability": float(adjusted_prob * 100),
            "is_phishing": is_phishing,
            "trust_indicators": trust_indicators,
            "warning_indicators": warning_indicators,
            "features": {k: str(float(v)) if isinstance(v, (int, float)) else str(v) 
                       for k, v in features.items()},
            "model_name": model_name
        }
        
    except Exception as e:
        logger.error(f"Error analyzing URL: {str(e)}")
        return {"error": str(e)}

def is_private_ip(ip):
    """Check if an IP address is private"""
    try:
        import ipaddress
        return ipaddress.ip_address(ip).is_private
    except:
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    url = data.get('url', '').strip()
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
        
    result = analyze_url(url)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

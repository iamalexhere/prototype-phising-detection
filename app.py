from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from phishing_detection import URLFeatureExtractor
import logging
from pathlib import Path
import json
from datetime import datetime
from screenshot import capture_website_screenshot

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

def get_risk_classification(probability):
    """
    Get detailed risk classification based on final adjusted probability score
    Returns dictionary with classification details
    """
    if probability < 0.2:
        return {
            "level": "Safe",
            "description": "This website has passed our security checks",
            "color": "success",
            "icon": "fa-check-circle",
            "details": "After analyzing all security factors including domain history, SSL certificate, and DNS records, this website shows strong signs of being legitimate"
        }
    elif probability < 0.4:
        return {
            "level": "Low Risk",
            "description": "This website appears mostly safe but has minor security concerns",
            "color": "info",
            "icon": "fa-info-circle",
            "details": "While the overall security score is good, there are some recommended security improvements that could be made"
        }
    elif probability < 0.6:
        return {
            "level": "Medium Risk",
            "description": "This website has some concerning security issues",
            "color": "warning",
            "icon": "fa-exclamation-circle",
            "details": "Our analysis found several security concerns. While not definitively malicious, exercise caution and verify the website's legitimacy before proceeding"
        }
    elif probability < 0.8:
        return {
            "level": "High Risk",
            "description": "Multiple security issues detected - likely malicious",
            "color": "danger",
            "icon": "fa-exclamation-triangle",
            "details": "This website exhibits many characteristics commonly associated with phishing attempts. It is strongly recommended to avoid entering any sensitive information"
        }
    else:
        return {
            "level": "Critical Risk",
            "description": "This website is almost certainly malicious",
            "color": "critical",
            "icon": "fa-skull-crossbones",
            "details": "Our analysis indicates this is very likely a phishing website. DO NOT proceed or enter any information. If you've already shared any data, consider it compromised"
        }

def analyze_url(url):
    """Analyze URL and return phishing prediction results"""
    try:
        # Load model
        model, feature_names, model_name = load_latest_model()
        logger.info(f"Using model: {model_name}")
        logger.info(f"Loaded {len(feature_names)} features: {', '.join(feature_names)}")
        
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
        phishing_prob = probability[1] * 100  # Convert to percentage
        
        # Try to capture screenshot
        screenshot_success, screenshot_path = capture_website_screenshot(url)
        
        # Calculate trust score
        trust_score = 0
        
        # Prepare insights
        insights = {
            'domain_health': {
                'title': 'Domain Health',
                'status': 'Good' if features.get('domain_age_days', 0) > 180 else 'Suspicious',
                'details': []
            },
            'dns_security': {
                'title': 'DNS Security',
                'status': 'Secure' if all([
                    features.get('has_mx_record', 0),
                    features.get('has_ns_record', 0),
                    features.get('num_ns_records', 0) >= 2
                ]) else 'Incomplete',
                'details': []
            },
            'ssl_status': {
                'title': 'SSL Security',
                'status': 'Secure' if all([
                    features.get('is_https', 0),
                    features.get('ssl_is_valid', 0),
                    features.get('ssl_days_valid', 0) > 90
                ]) else 'Insecure',
                'details': []
            }
        }
        
        # Domain Health Details
        domain_age = features.get('domain_age_days', 0)
        if domain_age > 0:
            age_years = domain_age / 365
            insights['domain_health']['details'].append(
                f"Domain age: {age_years:.1f} years" if age_years >= 1 else f"Domain age: {int(domain_age)} days"
            )
        if features.get('has_registrar'):
            insights['domain_health']['details'].append("Domain has valid registrar information")
        if features.get('days_to_expiration'):
            days_left = features.get('days_to_expiration')
            insights['domain_health']['details'].append(
                f"Domain expires in {int(days_left)} days"
            )
            
        # DNS Security Details
        if features.get('has_mx_record'):
            mx_count = features.get('num_mx_records', 0)
            insights['dns_security']['details'].append(
                f"Has {mx_count} mail server{'s' if mx_count > 1 else ''}"
            )
        if features.get('has_ns_record'):
            ns_count = features.get('num_ns_records', 0)
            insights['dns_security']['details'].append(
                f"Has {ns_count} name server{'s' if ns_count > 1 else ''}"
            )
        if features.get('has_a_record'):
            insights['dns_security']['details'].append("Domain resolves to valid IP")
            if features.get('is_private_ip'):
                insights['dns_security']['details'].append("WARNING: Resolves to private IP address")
        
        # SSL Security Details
        if features.get('is_https'):
            insights['ssl_status']['details'].append("Uses HTTPS encryption")
            if features.get('ssl_is_valid'):
                days_valid = features.get('ssl_days_valid', 0)
                insights['ssl_status']['details'].append(
                    f"Valid SSL certificate (expires in {int(days_valid)} days)"
                )
            else:
                insights['ssl_status']['details'].append("WARNING: Invalid SSL certificate")
        else:
            insights['ssl_status']['details'].append("WARNING: No HTTPS encryption")
        
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
        adjusted_prob = max(0.01, min(0.99, phishing_prob / 100 - trust_score))
        is_phishing = adjusted_prob > 0.45
        
        # Get risk classification based on final score
        risk_class = get_risk_classification(adjusted_prob)
        
        # Prepare trust indicators
        trust_indicators = []
        warning_indicators = []
        
        # SSL/HTTPS indicators
        if features.get('is_https', 0):
            trust_indicators.append("Uses HTTPS encryption")
            if features.get('ssl_is_valid', 0):
                trust_indicators.append(f"Valid SSL certificate (expires in {int(features.get('ssl_days_valid', 0))} days)")
            else:
                warning_indicators.append("Invalid SSL certificate")
        else:
            warning_indicators.append("No HTTPS encryption")
            
        # Domain age indicators
        domain_age = features.get('domain_age_days', 0)
        if domain_age > 365:
            trust_indicators.append(f"Domain is {domain_age/365:.1f} years old")
        elif domain_age > 180:
            trust_indicators.append(f"Domain is {int(domain_age)} days old")
        else:
            warning_indicators.append(f"Domain is only {int(domain_age)} days old")
            
        # DNS record indicators
        if features.get('has_mx_record', 0):
            trust_indicators.append(f"Has {features.get('num_mx_records', 0)} mail servers")
        if features.get('has_ns_record', 0):
            trust_indicators.append(f"Has {features.get('num_ns_records', 0)} name servers")
            
        # URL structure indicators
        if features.get('has_ip', 0):
            warning_indicators.append("Uses IP address instead of domain name")
        if features.get('has_at_symbol', 0):
            warning_indicators.append("Contains @ symbol in URL")
        if features.get('has_multiple_subdomains', 0):
            warning_indicators.append("Uses multiple subdomains")
        if features.get('has_double_slash', 0):
            warning_indicators.append("Contains double slash in path")
            
        return {
            "url": url,
            "raw_probability": float(phishing_prob),
            "trust_score": float(trust_score * 100),
            "adjusted_probability": float(adjusted_prob * 100),
            "is_phishing": bool(is_phishing),
            "risk_classification": risk_class,
            "trust_indicators": list(trust_indicators),
            "warning_indicators": list(warning_indicators),
            "features": {k: str(float(v)) if isinstance(v, (int, float)) else str(v) 
                       for k, v in features.items()},
            "model_name": str(model_name),
            "insights": dict(insights),
            "screenshot": str(screenshot_path) if screenshot_success else None
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

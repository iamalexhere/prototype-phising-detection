import argparse
import joblib
import logging
from pathlib import Path
from phishing_detection import URLFeatureExtractor
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def load_latest_model(models_dir='models/split_10'):
    """Load the latest trained model and its feature names"""
    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory {models_dir} not found")
    
    # Find model and feature names files
    model_files = list(models_path.glob("phishing_detector_split_9_*.joblib"))
    feature_files = list(models_path.glob("feature_names_split_9_*.joblib"))
    
    if not model_files or not feature_files:
        raise FileNotFoundError("Model or feature names file not found")
    
    # Get the latest files based on timestamp in filename
    latest_model = max(model_files, key=lambda x: x.name)
    latest_features = max(feature_files, key=lambda x: x.name)
    
    # Load the latest model and feature names
    model = joblib.load(latest_model)
    feature_names = joblib.load(latest_features)
    
    return model, feature_names, latest_model.name

def format_feature_value(value):
    """Format feature value for display"""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    elif isinstance(value, (int, float)):
        return f"{value:,.2f}" if isinstance(value, float) else str(value)
    return str(value)

def is_private_ip(ip):
    """Check if an IP address is private"""
    try:
        import ipaddress
        return ipaddress.ip_address(ip).is_private
    except:
        return False

def normalize_feature_vector(feature_vector, feature_names):
    """Normalize features to a common scale"""
    normalized = feature_vector.copy()
    
    # Normalize numeric features to [0,1] range
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

def main():
    parser = argparse.ArgumentParser(description='Predict if a URL is phishing using machine learning')
    parser.add_argument('url', help='URL to analyze')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed feature information')
    args = parser.parse_args()
    
    try:
        # Load the model and feature names
        logger.info("Loading model...")
        model, feature_names, model_filename = load_latest_model()
        
        # Initialize feature extractor
        extractor = URLFeatureExtractor()
        
        # Extract features
        logger.info("\nExtracting features...")
        features = extractor.extract_features(args.url)
        
        if features is None:
            logger.error("Failed to extract features from URL")
            return
        
        # Add is_private_ip feature
        if features.get('has_ip', 0):
            import re
            ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
            ip_match = re.search(ip_pattern, args.url)
            features['is_private_ip'] = is_private_ip(ip_match.group(0)) if ip_match else False
        else:
            features['is_private_ip'] = False
        
        # Display extracted features if verbose mode is on
        if args.verbose:
            logger.info("\nExtracted Features:")
            logger.info("-" * 50)
            for feature, value in sorted(features.items()):
                if feature in feature_names:
                    logger.info(f"  {feature.replace('_', ' ').title()}: {value}")
        
        # Prepare features for prediction
        feature_dict = {name: 0 for name in feature_names}  # Initialize all features to 0
        feature_dict.update(features)  # Update with extracted features
        
        # Create DataFrame with all required features
        feature_vector = pd.DataFrame([feature_dict])[feature_names]
        
        # Normalize features
        feature_vector = normalize_feature_vector(feature_vector, feature_names)
        
        # Display feature vector if verbose
        if args.verbose:
            logger.info("\nFeature Vector (after normalization):")
            logger.info("-" * 50)
            for feature_name in feature_names:
                value = feature_vector[feature_name].iloc[0]
                logger.info(f"{feature_name}: {value}")
        
        # Get prediction and probability
        probability = model.predict_proba(feature_vector)[0]
        phishing_prob = probability[1]  # Probability of being phishing
        
        # Calculate trust score based on normalized features
        trust_score = 0
        
        # SSL Group (~50% importance) - using normalized values
        if features.get('is_https', 0) and features.get('ssl_is_valid', 0):
            trust_score += 0.25
            normalized_ssl_days = feature_vector['ssl_days_valid'].iloc[0] if 'ssl_days_valid' in feature_names else 0
            if normalized_ssl_days > 0.5:  # If normalized value > 0.5 (relatively long validity)
                trust_score += 0.15
        
        # URL Group (~29% importance) - using normalized values
        normalized_domain_length = feature_vector['domain_length'].iloc[0] if 'domain_length' in feature_names else 1
        if not features.get('has_multiple_subdomains', 0) and normalized_domain_length < 0.5:
            trust_score += 0.15
            
        # DNS Group (~11% importance) - using normalized values
        normalized_domain_age = feature_vector['domain_age_days'].iloc[0] if 'domain_age_days' in feature_names else 0
        if normalized_domain_age > 0.7:  # If normalized age is relatively high
            trust_score += 0.10
        if features.get('has_mx_record', 0) and features.get('has_ns_record', 0):
            trust_score += 0.10

        # Calculate adjusted probability
        adjusted_prob = max(0.01, min(0.99, phishing_prob - trust_score))
        
        # Use optimal threshold from training (average ~0.45)
        is_phishing = adjusted_prob > 0.45
        
        # Display result
        logger.info("\nPrediction Results:")
        logger.info("-" * 50)
        logger.info(f"Model Used: {model_filename}")
        logger.info(f"Raw Probability: {phishing_prob:.2%}")
        logger.info(f"Trust Score Adjustment: -{trust_score:.2%}")
        logger.info(f"Adjusted Probability: {adjusted_prob:.2%}")
        logger.info(f"Verdict: {'PHISHING' if is_phishing else 'LEGITIMATE'}")
        
        if trust_score > 0:
            logger.info("\nTrust Indicators:")
            if features.get('is_https', 0) and features.get('ssl_is_valid', 0):
                logger.info(f"- Valid HTTPS/SSL certificate (valid for {features.get('ssl_days_valid', 0)} days)")
            if not features.get('has_multiple_subdomains', 0):
                logger.info("- Simple domain structure (no multiple subdomains)")
            if features.get('domain_age_days', 0) > 365:
                logger.info(f"- Domain age: {features['domain_age_days'] / 365:.1f} years")
            if features.get('has_mx_record', 0) and features.get('has_ns_record', 0):
                logger.info("- Valid DNS records (MX and NS)")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()

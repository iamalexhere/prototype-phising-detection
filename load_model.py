import os
import joblib
from datetime import datetime
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def find_latest_model(base_dir='models'):
    """Find the latest model and its corresponding feature names file."""
    latest_model = None
    latest_features = None
    latest_time = datetime.min
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith('phishing_detector_') and file.endswith('.joblib'):
                try:
                    # Extract timestamp from filename
                    # Handle both formats:
                    # 1. phishing_detector_YYYYMMDD_HHMMSS.joblib
                    # 2. phishing_detector_split_N_YYYYMMDD_HHMMSS.joblib
                    parts = file.replace('phishing_detector_', '').replace('.joblib', '').split('_')
                    
                    # If it's a split model, get the last two parts for timestamp
                    if 'split' in parts:
                        timestamp_str = f"{parts[-2]}_{parts[-1]}"
                    else:
                        timestamp_str = f"{parts[0]}_{parts[1]}"
                        
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    
                    if timestamp > latest_time:
                        latest_time = timestamp
                        latest_model = os.path.join(root, file)
                        # Look for corresponding feature names file
                        feature_file = os.path.join(root, f"feature_names_{timestamp_str}.joblib")
                        if os.path.exists(feature_file):
                            latest_features = feature_file
                except Exception as e:
                    logging.warning(f"Skipping file {file}: {str(e)}")
                    continue
    
    if not latest_model or not latest_features:
        logging.error("Could not find valid model and feature files!")
        return None, None
        
    logging.info(f"Found latest model: {latest_model}")
    logging.info(f"Found latest features: {latest_features}")
    return latest_model, latest_features

def main():
    setup_logging()
    logging.info("Looking for the latest trained model...")
    
    latest_model_path, latest_features_path = find_latest_model()
    
    if not latest_model_path or not latest_features_path:
        logging.error("Could not find matching model and feature files!")
        return
    
    logging.info(f"Found latest model: {latest_model_path}")
    logging.info(f"Found latest features: {latest_features_path}")
    
    try:
        # Load the model and features
        model = joblib.load(latest_model_path)
        feature_names = joblib.load(latest_features_path)
        
        # Print model information
        logging.info("\nModel Information:")
        logging.info(f"Model type: {type(model).__name__}")
        
        if hasattr(model, 'get_params'):
            logging.info("\nModel Parameters:")
            for param, value in model.get_params().items():
                logging.info(f"{param}: {value}")
        
        logging.info(f"\nNumber of features: {len(feature_names)}")
        logging.info("\nFeature names:")
        for i, feature in enumerate(feature_names, 1):
            logging.info(f"{i}. {feature}")
            
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    main()

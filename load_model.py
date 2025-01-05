import os
import joblib
from datetime import datetime
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def find_latest_model(base_dir='models/split_10'):
    """Find the latest model and its corresponding feature names file."""
    logging.info("Searching for latest model in %s", base_dir)
    latest_model = None
    latest_features = None
    latest_time = datetime.min
    
    # Ensure base directory exists
    if not os.path.exists(base_dir):
        logging.error(f"Base directory {base_dir} does not exist!")
        return None, None
    
    # List all files in directory
    files = os.listdir(base_dir)
    
    # Find model files and their timestamps
    for file in files:
        if file.startswith('phishing_detector_split_9_') and file.endswith('.joblib'):
            try:
                # Extract timestamp from filename
                # Format: phishing_detector_split_9_YYYYMMDD_HHMMSS.joblib
                timestamp_str = '_'.join(file.split('_')[-2:]).replace('.joblib', '')
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                if timestamp > latest_time:
                    latest_time = timestamp
                    latest_model = os.path.join(base_dir, file)
                    # Look for corresponding feature names file
                    feature_file = os.path.join(base_dir, f"feature_names_split_9_{timestamp_str}.joblib")
                    if os.path.exists(feature_file):
                        latest_features = feature_file
                        logging.info(f"Found newer model from {timestamp_str}")
            except Exception as e:
                logging.warning(f"Skipping file {file}: {str(e)}")
                continue
    
    if not latest_model or not latest_features:
        logging.error("Could not find valid model and feature files!")
        return None, None
    
    logging.info(f"Latest model: {os.path.basename(latest_model)}")
    logging.info(f"Latest features: {os.path.basename(latest_features)}")
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

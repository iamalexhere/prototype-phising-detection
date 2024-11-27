import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import ipaddress
import joblib
import tldextract
from datetime import datetime

class URLFeatureExtractor:
    """
    A class to extract features from URLs for phishing detection
    """
    
    @staticmethod
    def is_ip_address(url):
        """Check if the URL uses an IP address instead of a domain name."""
        try:
            domain = urlparse(url).netloc
            ipaddress.ip_address(domain)
            return 1
        except:
            return 0
    
    @staticmethod
    def extract_features(url):
        """
        Extract features from a given URL
        
        Features:
        - URL length
        - Domain length
        - Path length
        - Number of special characters
        - Number of digits
        - Presence of @ symbol
        - Use of IP address
        - Additional security features
        """
        try:
            parsed = urlparse(url)
            extracted = tldextract.extract(url)
            
            # Length-based features
            url_length = len(url)
            domain_length = len(parsed.netloc)
            path_length = len(parsed.path)
            
            # Character-based features
            special_chars = len(re.findall(r'[^a-zA-Z0-9]', url))
            digits = len(re.findall(r'\d', url))
            has_at_symbol = '@' in url
            is_ip = URLFeatureExtractor.is_ip_address(url)
            
            # Additional security features
            num_dots = url.count('.')
            num_hyphens = url.count('-')
            num_underscores = url.count('_')
            num_percent = url.count('%')
            num_query_components = len(parsed.query.split('&')) if parsed.query else 0
            num_ampersand = url.count('&')
            num_hash = url.count('#')
            has_https = int(parsed.scheme == 'https')
            
            # Domain-based features
            domain_token_count = len(re.findall(r'[a-zA-Z0-9]+', extracted.domain))
            subdomain_length = len(extracted.subdomain)
            tld_length = len(extracted.suffix) if extracted.suffix else 0
            
            return {
                'url_length': url_length,
                'domain_length': domain_length,
                'path_length': path_length,
                'special_chars_count': special_chars,
                'digits_count': digits,
                'has_at_symbol': int(has_at_symbol),
                'is_ip_address': is_ip,
                'num_dots': num_dots,
                'num_hyphens': num_hyphens,
                'num_underscores': num_underscores,
                'num_percent': num_percent,
                'num_query_components': num_query_components,
                'num_ampersand': num_ampersand,
                'num_hash': num_hash,
                'has_https': has_https,
                'domain_token_count': domain_token_count,
                'subdomain_length': subdomain_length,
                'tld_length': tld_length
            }
        except Exception as e:
            print(f"Error processing URL: {url}")
            print(f"Error message: {str(e)}")
            return None

def load_and_process_data(phishing_file_path, legitimate_file_path):
    """
    Load and process both phishing and legitimate URL datasets
    """
    # Load the datasets
    print("Loading datasets...")
    phishing_df = pd.read_csv(phishing_file_path, low_memory=False)
    legitimate_df = pd.read_csv(legitimate_file_path, low_memory=False)
    
    print(f"Total phishing URLs: {len(phishing_df)}")
    print(f"Total legitimate URLs: {len(legitimate_df)}")
    
    # Sample equal number of URLs from both datasets to create a balanced dataset
    min_size = min(len(phishing_df), len(legitimate_df))
    phishing_df = phishing_df.sample(n=min_size, random_state=42)
    legitimate_df = legitimate_df.sample(n=min_size, random_state=42)
    
    # Create feature lists for both types
    print("\nExtracting features from URLs...")
    
    def process_urls(urls, is_phishing):
        features_list = []
        labels = []
        for url in urls:
            features = URLFeatureExtractor.extract_features(url)
            if features:
                features_list.append(features)
                labels.append(1 if is_phishing else 0)
        return features_list, labels
    
    # Process phishing URLs
    phishing_features, phishing_labels = process_urls(phishing_df['url'], True)
    print(f"Processed {len(phishing_features)} phishing URLs")
    
    # Process legitimate URLs
    legitimate_features, legitimate_labels = process_urls(legitimate_df['url'], False)
    print(f"Processed {len(legitimate_features)} legitimate URLs")
    
    # Combine features and labels
    all_features = phishing_features + legitimate_features
    all_labels = phishing_labels + legitimate_labels
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    features_df['label'] = all_labels
    
    print("\nFeature statistics:")
    print(features_df.describe())
    
    print("\nLabel distribution:")
    print(features_df['label'].value_counts(normalize=True))
    
    return features_df

def prepare_data_splits(features_df, test_size=0.3, val_size=0.15):
    """
    Split data into train, validation, and test sets (70-15-15)
    """
    # First split: 70% train, 30% temp (which will be split into validation and test)
    X = features_df.drop('label', axis=1)
    y = features_df['label']
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: Split temp into validation and test (50% each of the 30%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print("\nData split sizes:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def tune_random_forest(X_train, y_train):
    """
    Perform hyperparameter tuning for Random Forest using GridSearchCV
    """
    print("\nStarting Random Forest hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1',
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def save_model(model, feature_names, output_dir='models'):
    """
    Save the trained model and feature names
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate timestamp for versioning
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the model
    model_path = os.path.join(output_dir, f'phishing_detector_{timestamp}.joblib')
    joblib.dump(model, model_path)
    
    # Save feature names
    feature_names_path = os.path.join(output_dir, f'feature_names_{timestamp}.joblib')
    joblib.dump(feature_names, feature_names_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Feature names saved to: {feature_names_path}")
    
    return model_path, feature_names_path

def main():
    # File paths
    phishing_file_path = "verified_online.csv"
    legitimate_file_path = "URL-categorization-DFE.csv"
    
    # Load and process data
    print("Starting phishing URL detection model training...")
    features_df = load_and_process_data(phishing_file_path, legitimate_file_path)
    
    # Split the data
    print("\nSplitting data into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(features_df)
    
    # Perform hyperparameter tuning
    best_model = tune_random_forest(X_train, y_train)
    
    # Evaluate on validation set
    print("\nValidation Set Performance:")
    y_val_pred = best_model.predict(X_val)
    print(classification_report(y_val, y_val_pred))
    
    # Final evaluation on test set
    print("\nTest Set Performance:")
    y_test_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_test_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_test_pred))
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save the model and feature names
    model_path, feature_names_path = save_model(best_model, X_train.columns.tolist())
    
    # Print example usage
    print("\nExample usage in your web app:")
    print("""
    import joblib
    
    # Load the model and feature names
    model = joblib.load('path_to_model.joblib')
    feature_names = joblib.load('path_to_feature_names.joblib')
    
    # Create feature extractor
    extractor = URLFeatureExtractor()
    
    def predict_url(url):
        # Extract features
        features = extractor.extract_features(url)
        if features is None:
            return None
            
        # Convert to DataFrame with correct feature order
        features_df = pd.DataFrame([features])[feature_names]
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0]
        
        return {
            'is_phishing': bool(prediction),
            'confidence': float(probability[1])
        }
    """)

if __name__ == "__main__":
    main()
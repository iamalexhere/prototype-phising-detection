import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re
from sklearn.model_selection import (
    train_test_split, learning_curve, StratifiedKFold,
    cross_val_score, cross_validate
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    log_loss, brier_score_loss
)
from sklearn.model_selection import GridSearchCV
import ipaddress
import joblib
import tldextract
from datetime import datetime, date
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
import whois
import socket
from bs4 import BeautifulSoup
import requests
from googlesearch import search
import urllib3
import dns.resolver
from dns_features import extract_dns_features
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
from typing import List, Dict, Any
import time

# Configure logging
def setup_logging(log_dir='logs'):
    """Setup logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'phishing_detection_{timestamp}.log')
    
    # Create formatters and handlers
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Rotating file handler (max 10MB per file, keep 5 backup files)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"Logging setup complete. Log file: {log_file}")
    return log_file

urllib3.disable_warnings()
warnings.filterwarnings('ignore', category=UserWarning)

# Configure matplotlib for non-interactive backend
plt.ioff()

# Create connection pool manager with retry strategy
session = requests.Session()
retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retries)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Thread-local storage for session reuse
thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        thread_local.session.mount('http://', adapter)
        thread_local.session.mount('https://', adapter)
    return thread_local.session

# Cache for DNS and WHOIS lookups
@lru_cache(maxsize=1024)
def cached_dns_lookup(domain):
    try:
        return dns.resolver.resolve(domain, 'A')
    except:
        return None

@lru_cache(maxsize=1024)
def cached_whois_lookup(domain):
    try:
        return whois.whois(domain)
    except:
        return None

class URLFeatureExtractor:
    """
    A class to extract features from URLs for phishing detection with optimized parallel processing
    """
    
    def __init__(self, max_workers=10):
        self.max_workers = max_workers
        self.feature_cache = {}
    
    @staticmethod
    def is_ip_address(url):
        """Check if the URL uses an IP address instead of a domain name."""
        try:
            domain = urlparse(url).netloc
            ipaddress.ip_address(domain)
            return 1
        except:
            return 0
    
    def extract_features_batch(self, urls: List[str], batch_size=100) -> pd.DataFrame:
        """
        Extract features from a batch of URLs using parallel processing
        """
        all_features = []
        
        # Process URLs in batches
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_url = {executor.submit(self.extract_features, url): url 
                               for url in batch_urls if url not in self.feature_cache}
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        features = future.result()
                        self.feature_cache[url] = features
                        all_features.append(features)
                    except Exception as e:
                        logging.error(f"Error extracting features for {url}: {str(e)}")
                        continue
            
            # Add cached features
            cached_features = [self.feature_cache[url] for url in batch_urls 
                             if url in self.feature_cache]
            all_features.extend(cached_features)
            
            # Log progress
            logging.info(f"Processed {i + len(batch_urls)}/{len(urls)} URLs")
        
        # Convert to DataFrame efficiently
        return pd.DataFrame(all_features)
    
    def extract_features(self, url: str) -> Dict[str, Any]:
        """
        Extract features from a given URL including DNS and domain-based features
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # Extract DNS features using our improved dns_features module
            dns_features_df = extract_dns_features(url)
            # Convert Series to scalar values
            dns_features = {col: dns_features_df[col].iloc[0] for col in dns_features_df.columns}
            
            # Basic URL features
            features = {
                'url_length': len(url),
                'domain_length': len(domain),
                'has_ip': self.is_ip_address(url),
                'has_at_symbol': '@' in url,
                'has_double_slash': '//' in parsed.path,
                'has_dash': '-' in domain,
                'has_multiple_subdomains': len(domain.split('.')) > 2,
                'is_https': parsed.scheme == 'https',
            }
            
            # Add DNS features
            features.update({
                'has_a_record': dns_features['has_a_record'],
                'num_a_records': dns_features['num_a_records'],
                'is_private_ip': dns_features['is_private_ip'],
                'has_mx_record': dns_features['has_mx_record'],
                'num_mx_records': dns_features['num_mx_records'],
                'has_ns_record': dns_features['has_ns_record'],
                'num_ns_records': dns_features['num_ns_records'],
                'domain_age_days': dns_features['domain_age_days'],
                'is_domain_young': dns_features['is_domain_young'],
                'days_to_expiration': dns_features['days_to_expiration'],
                'is_expiring_soon': dns_features['is_expiring_soon'],
                'has_registrar': dns_features['has_registrar'],
                'has_registrant': dns_features['has_registrant'],
                'ssl_days_valid': dns_features['ssl_days_valid'],
                'ssl_is_valid': dns_features['ssl_is_valid'],
                'ssl_is_expired': dns_features['ssl_is_expired'],
                'ssl_is_self_signed': dns_features['ssl_is_self_signed']
            })
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features for {url}: {str(e)}")
            return None
    
    @staticmethod
    def _get_domain_age(whois_info):
        """Calculate domain age in days"""
        if not whois_info or not whois_info.creation_date:
            return -1
        
        if isinstance(whois_info.creation_date, list):
            creation_date = whois_info.creation_date[0]
        else:
            creation_date = whois_info.creation_date
            
        if isinstance(creation_date, str):
            try:
                creation_date = datetime.strptime(creation_date, '%Y-%m-%d')
            except:
                return -1
                
        return (datetime.now() - creation_date).days

def load_and_process_data(phishing_file_path, legitimate_file_path, sample_size=100):
    """
    Load and process both phishing and legitimate URL datasets with enhanced preprocessing
    
    Parameters:
    - phishing_file_path: path to phishing URLs file
    - legitimate_file_path: path to legitimate URLs file
    - sample_size: number of samples per class (default increased to 50000)
    """
    logging.info("Loading and processing datasets...")
    
    # Load datasets
    phishing_df = pd.read_csv(phishing_file_path)
    legitimate_df = pd.read_csv(legitimate_file_path)
    
    # Ensure URL column exists
    phishing_urls = phishing_df['url'] if 'url' in phishing_df.columns else phishing_df.iloc[:, 0]
    legitimate_urls = legitimate_df['url'] if 'url' in legitimate_df.columns else legitimate_df.iloc[:, 0]
    
    # Sample equal numbers from each class
    if len(phishing_urls) > sample_size:
        phishing_urls = phishing_urls.sample(n=sample_size, random_state=42)
    if len(legitimate_urls) > sample_size:
        legitimate_urls = legitimate_urls.sample(n=sample_size, random_state=42)
    
    logging.info(f"Processing {len(phishing_urls)} phishing and {len(legitimate_urls)} legitimate URLs")
    
    # Initialize feature extractor
    extractor = URLFeatureExtractor()
    
    # Extract features in parallel
    phishing_features = extractor.extract_features_batch(phishing_urls.tolist())
    legitimate_features = extractor.extract_features_batch(legitimate_urls.tolist())
    
    # Convert to DataFrames
    phishing_df = pd.DataFrame(phishing_features)
    legitimate_df = pd.DataFrame(legitimate_features)
    
    # Add labels
    phishing_df['label'] = 1
    legitimate_df['label'] = 0
    
    # Combine datasets
    features_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)
    
    # Preprocessing steps
    # 1. Handle missing values
    features_df = features_df.fillna(features_df.mean())
    
    # 2. Remove constant features
    constant_features = [col for col in features_df.columns if col != 'label' 
                        and features_df[col].nunique() == 1]
    features_df = features_df.drop(columns=constant_features)
    if constant_features:
        logging.info(f"Removed {len(constant_features)} constant features")
    
    # 3. Remove highly correlated features
    corr_matrix = features_df.drop('label', axis=1).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper.columns if any(upper[column] > 0.95)]
    features_df = features_df.drop(columns=high_corr_features)
    if high_corr_features:
        logging.info(f"Removed {len(high_corr_features)} highly correlated features")
    
    # 4. Scale numerical features
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    numerical_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = numerical_cols.drop('label') if 'label' in numerical_cols else numerical_cols
    
    if len(numerical_cols) > 0:
        features_df[numerical_cols] = scaler.fit_transform(features_df[numerical_cols])
        logging.info(f"Scaled {len(numerical_cols)} numerical features using RobustScaler")
    
    # Shuffle the dataset
    features_df = features_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logging.info(f"Final dataset shape: {features_df.shape}")
    return features_df

def extract_domain(url):
    """Extract the base domain from a URL"""
    try:
        ext = tldextract.extract(url)
        return f"{ext.domain}.{ext.suffix}"
    except:
        return None

def prepare_data_splits(features_df, test_size=0.2, val_size=0.2):
    """
    Split data into train, validation, and test sets using domain-based splitting
    to prevent domain leakage between sets
    
    Parameters:
    - features_df: DataFrame containing features and labels
    - test_size: proportion of data for test set
    - val_size: proportion of data for validation set
    
    Returns:
    - X_train, X_val, X_test: feature matrices
    - y_train, y_val, y_test: target vectors
    - feature_names: list of feature names
    """
    logging.info("\nSplitting data into train, validation, and test sets...")
    
    # Extract domains from URLs
    domains = features_df['url'].apply(extract_domain)
    unique_domains = domains.unique()
    
    # Split domains into train, validation, and test sets
    n_domains = len(unique_domains)
    n_test = int(n_domains * test_size)
    n_val = int(n_domains * val_size)
    
    # Randomly shuffle domains
    np.random.seed(42)
    shuffled_domains = np.random.permutation(unique_domains)
    
    test_domains = set(shuffled_domains[:n_test])
    val_domains = set(shuffled_domains[n_test:n_test + n_val])
    train_domains = set(shuffled_domains[n_test + n_val:])
    
    # Split data based on domains
    test_mask = domains.isin(test_domains)
    val_mask = domains.isin(val_domains)
    train_mask = domains.isin(train_domains)
    
    # Get feature names (excluding 'url' and 'label' columns)
    feature_names = [col for col in features_df.columns if col not in ['url', 'label']]
    
    # Split features and labels
    X = features_df[feature_names]
    y = features_df['label']
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # Log data split information
    logging.info("Data split sizes and class distribution:")
    logging.info(f"Total dataset size: {len(features_df)} samples")
    logging.info(f"Training set: {len(X_train)} samples ({len(X_train)/len(features_df)*100:.1f}%)")
    logging.info(f"  - Class distribution: {np.bincount(y_train)}")
    logging.info(f"  - Number of domains: {len(train_domains)}")
    logging.info(f"Validation set: {len(X_val)} samples ({len(X_val)/len(features_df)*100:.1f}%)")
    logging.info(f"  - Class distribution: {np.bincount(y_val)}")
    logging.info(f"  - Number of domains: {len(val_domains)}")
    logging.info(f"Test set: {len(X_test)} samples ({len(X_test)/len(features_df)*100:.1f}%)")
    logging.info(f"  - Class distribution: {np.bincount(y_test)}")
    logging.info(f"  - Number of domains: {len(test_domains)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

def tune_random_forest(X_train, y_train):
    """
    Perform hyperparameter tuning for Random Forest using GridSearchCV with enhanced cross-validation
    and regularization parameters
    """
    logging.info("Starting hyperparameter tuning for Random Forest...")
    
    # Define parameter grid with regularization parameters
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'max_samples': [0.8, 1.0],  # Bootstrap sample size
        'class_weight': ['balanced'],
        'ccp_alpha': [0.0, 0.01, 0.02]  # Pruning parameter
    }
    
    # Initialize base classifier
    base_clf = RandomForestClassifier(random_state=42)
    
    # Use StratifiedKFold with shuffling for better cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize GridSearchCV with multiple scoring metrics
    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        cv=cv,
        scoring={
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        },
        refit='f1',  # Use F1 score for selecting best model
        n_jobs=-1,
        verbose=1
    )
    
    # Fit GridSearchCV
    logging.info("Fitting GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    # Get best parameters and scores
    logging.info("\nBest parameters found:")
    logging.info(grid_search.best_params_)
    logging.info("\nBest cross-validation scores:")
    for metric, score in grid_search.cv_results_['mean_test_' + grid_search.refit].items():
        logging.info(f"{metric}: {score:.4f}")
    
    # Calibrate probabilities using the best model
    best_model = grid_search.best_estimator_
    calibrated_model = CalibratedClassifierCV(
        best_model, 
        cv='prefit',
        method='sigmoid'
    )
    calibrated_model.fit(X_train, y_train)
    
    return calibrated_model

def evaluate_model(model, X, y, set_name=""):
    """
    Comprehensive model evaluation function with threshold optimization
    """
    # Get predicted probabilities
    y_prob = model.predict_proba(X)[:, 1]
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Apply optimal threshold
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    # Calculate metrics
    logging.info(f"\n{set_name} Set Performance:")
    logging.info(classification_report(y, y_pred))
    
    logging.info(f"\nDetailed {set_name} Set Metrics:")
    logging.info(f"Brier Score: {brier_score_loss(y, y_prob):.4f}")
    logging.info(f"Log Loss: {log_loss(y, y_prob):.4f}")
    logging.info(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    return y_pred, y_prob, optimal_threshold

class ModelVisualizer:
    """
    A class to generate and save various model evaluation plots
    """
    def __init__(self, output_dir='plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('default')  # Using default style instead of seaborn
    
    def save_plot(self, plot_name):
        """Save the current plot to the output directory"""
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{plot_name}.png", dpi=300, bbox_inches='tight')
        plt.close('all')  # Properly close all figures

    def plot_confusion_matrix(self, y_true, y_pred, classes=['Legitimate', 'Phishing']):
        """Generate and save confusion matrix plot"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        self.save_plot('confusion_matrix-100')

    def plot_feature_importance(self, feature_names, importances):
        """Generate and save feature importance plot"""
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        sns.barplot(data=importance_df, y='feature', x='importance')
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        self.save_plot('feature_importance-100')

    def plot_roc_curve(self, y_true, y_prob):
        """Generate and save ROC curve plot"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        self.save_plot('roc_curve-100')

    def plot_precision_recall_curve(self, y_true, y_prob):
        """Generate and save precision-recall curve plot"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        self.save_plot('precision_recall_curve-100')

    def plot_learning_curve(self, estimator, X, y, cv=5):
        """Generate and save learning curve plot"""
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.3, 1.0, 5),
            scoring='f1'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score',
                color='darkorange', lw=2)
        plt.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1,
                        color='darkorange')
        plt.plot(train_sizes, test_mean, label='Cross-validation score',
                color='navy', lw=2)
        plt.fill_between(train_sizes, test_mean - test_std,
                        test_mean + test_std, alpha=0.1,
                        color='navy')
        
        plt.xlabel('Training Examples')
        plt.ylabel('F1 Score')
        plt.title('Learning Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        self.save_plot('learning_curve-100')

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
    
    logging.info(f"\nModel saved to: {model_path}")
    logging.info(f"Feature names saved to: {feature_names_path}")
    
    return model_path, feature_names_path

def main():
    try:
        # Setup logging
        log_file = setup_logging()
        
        # File paths
        phishing_file_path = "verified_online.csv"
        legitimate_file_path = "URL-categorization-DFE.csv"
        
        # Initialize visualizer
        visualizer = ModelVisualizer()
        
        # Load and process data
        logging.info("Starting phishing URL detection model training...")
        features_df = load_and_process_data(phishing_file_path, legitimate_file_path, sample_size=100)
        
        # Split the data
        logging.info("\nSplitting data into train, validation, and test sets...")
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data_splits(features_df)
        
        # Perform hyperparameter tuning with enhanced cross-validation
        best_model = tune_random_forest(X_train, y_train)
        
        # Generate learning curve plot
        logging.info("\nGenerating learning curve plot...")
        visualizer.plot_learning_curve(best_model, X_train, y_train)
        
        # Evaluate on validation set
        logging.info("\nEvaluating on validation set:")
        y_val_pred, y_val_prob, _ = evaluate_model(best_model, X_val, y_val, "Validation Set")
        
        # Generate validation set plots
        visualizer.plot_confusion_matrix(y_val, y_val_pred)
        visualizer.plot_roc_curve(y_val, y_val_prob)
        visualizer.plot_precision_recall_curve(y_val, y_val_prob)
        
        # Final evaluation on test set
        logging.info("\nEvaluating on test set:")
        y_test_pred, y_test_prob, _ = evaluate_model(best_model, X_test, y_test, "Test Set")
        
        # Generate test set plots
        logging.info("\nGenerating evaluation plots...")
        visualizer.plot_confusion_matrix(y_test, y_test_pred)
        visualizer.plot_roc_curve(y_test, y_test_prob)
        visualizer.plot_precision_recall_curve(y_test, y_test_prob)
        
        # Feature importance analysis and plot
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info("\nFeature Importance:")
        logging.info(feature_importance)
        visualizer.plot_feature_importance(
            feature_importance['feature'].values,
            feature_importance['importance'].values
        )
        
        # Save the model and feature names
        model_path, feature_names_path = save_model(best_model, feature_names)

    finally:
        # Cleanup
        plt.close('all')
        
if __name__ == "__main__":
    main()
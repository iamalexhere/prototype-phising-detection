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
    log_loss, brier_score_loss, accuracy_score
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
import pickle

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
    
    def extract_features_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Extract features from a batch of URLs using parallel processing
        Returns a list of feature dictionaries
        """
        features_list = []
        
        # Process URLs in batches
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for url in urls:
                if url in self.feature_cache:
                    features_list.append(self.feature_cache[url])
                else:
                    futures.append(executor.submit(self.extract_features, url))
            
            for future in as_completed(futures):
                try:
                    features = future.result()
                    features_list.append(features)
                except Exception as e:
                    logging.error(f"Error extracting features: {str(e)}")
                    # Add empty features for failed URLs
                    features_list.append({})
        
        return features_list
    
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

def extract_domain(url):
    """Extract the base domain from a URL"""
    try:
        ext = tldextract.extract(url)
        if not ext.domain or not ext.suffix:
            return None
        return f"{ext.domain}.{ext.suffix}"
    except:
        return None

def load_and_process_data(phishing_file_path: str, legitimate_file_path: str, sample_size: int = 100, batch_size: int = 50) -> pd.DataFrame:
    """
    Load and process both phishing and legitimate URL datasets with enhanced preprocessing
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
    
    # Add URLs to feature dictionaries
    for features, url in zip(phishing_features, phishing_urls):
        if isinstance(features, dict):
            features['url'] = url
    
    for features, url in zip(legitimate_features, legitimate_urls):
        if isinstance(features, dict):
            features['url'] = url
    
    # Convert to DataFrames
    phishing_df = pd.DataFrame(phishing_features)
    legitimate_df = pd.DataFrame(legitimate_features)
    
    # Add labels
    phishing_df['label'] = 1
    legitimate_df['label'] = 0
    
    # Combine datasets
    features_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)
    
    # Store URL column separately before preprocessing
    urls = features_df['url'].copy()
    
    # Remove URL column temporarily for preprocessing
    features_df_prep = features_df.drop('url', axis=1)
    
    # Preprocessing steps
    # 1. Handle missing values
    features_df_prep = features_df_prep.fillna(features_df_prep.mean())
    
    # 2. Remove constant features
    constant_features = [col for col in features_df_prep.columns if col != 'label' 
                        and features_df_prep[col].nunique() == 1]
    features_df_prep = features_df_prep.drop(columns=constant_features)
    if constant_features:
        logging.info(f"Removed {len(constant_features)} constant features")
    
    # 3. Remove highly correlated features
    corr_matrix = features_df_prep.drop('label', axis=1).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper.columns if any(upper[column] > 0.95)]
    features_df_prep = features_df_prep.drop(columns=high_corr_features)
    if high_corr_features:
        logging.info(f"Removed {len(high_corr_features)} highly correlated features")
    
    # 4. Scale numerical features
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    numerical_cols = features_df_prep.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = numerical_cols.drop('label') if 'label' in numerical_cols else numerical_cols
    
    if len(numerical_cols) > 0:
        features_df_prep[numerical_cols] = scaler.fit_transform(features_df_prep[numerical_cols])
        logging.info(f"Scaled {len(numerical_cols)} numerical features using RobustScaler")
    
    # Add URL column back
    features_df_prep['url'] = urls
    
    # Shuffle the dataset
    features_df_prep = features_df_prep.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logging.info(f"Final dataset shape: {features_df_prep.shape}")
    return features_df_prep

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
    and regularization parameters. Returns both the base model and calibrated model.
    """
    logging.info("Starting hyperparameter tuning for Random Forest...")
    
    # Define parameter grid with regularization parameters
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'],
        'max_samples': [0.8],  # Bootstrap sample size
        'class_weight': ['balanced'],
        'ccp_alpha': [0.0, 0.01]  # Pruning parameter
    }
    
    # Initialize base classifier with n_jobs for parallel processing
    base_clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
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
    
    # Log all metric scores
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        score = grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_]
        logging.info(f"{metric}: {score:.4f}")
    
    # Get the best base model
    best_base_model = grid_search.best_estimator_
    
    # Calibrate probabilities using the best model
    calibrated_model = CalibratedClassifierCV(
        RandomForestClassifier(**grid_search.best_params_, random_state=42), 
        method='sigmoid',
        cv=5  # Reduced CV folds for smaller dataset
    )
    calibrated_model.fit(X_train, y_train)
    
    return best_base_model, calibrated_model

class ModelVisualizer:
    """Class for visualizing model performance metrics"""
    
    def __init__(self, output_dir='plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_learning_curve(self, model, X, y):
        """
        Plot learning curve to visualize model's performance with varying training set sizes
        """
        logging.info("\nGenerating learning curve plot...")
        
        # Adjust train sizes for larger dataset
        train_sizes = np.linspace(0.2, 1.0, 3)  # Fewer points for learning curve
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        plt.figure(figsize=(12, 8))  # Larger figure for better visibility
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            n_jobs=-1,
            scoring='f1'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, val_mean, label='Cross-validation score', color='green', marker='o')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='green')
        
        plt.xlabel('Training Examples')
        plt.ylabel('F1 Score')
        plt.title('Learning Curve (500 Samples)')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        plt.savefig(os.path.join(self.output_dir, 'learning_curve-500.png'))
        plt.close()

    def save_plot(self, plot_name):
        """Save the current plot to the output directory"""
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{plot_name}.png"), dpi=300, bbox_inches='tight')
        plt.close('all')  # Properly close all figures

    def plot_confusion_matrix(self, y_true, y_pred, classes=['Legitimate', 'Phishing']):
        """Generate and save confusion matrix plot"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix (500 Samples)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        self.save_plot('confusion_matrix-500')

    def plot_feature_importance(self, feature_names, importances):
        """Generate and save feature importance plot"""
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        sns.barplot(data=importance_df, y='feature', x='importance')
        plt.title('Feature Importance (500 Samples)')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        self.save_plot('feature_importance-500')

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
        plt.title('Receiver Operating Characteristic (ROC) Curve (500 Samples)')
        plt.legend(loc="lower right")
        self.save_plot('roc_curve-500')

    def plot_precision_recall_curve(self, y_true, y_prob):
        """Generate and save precision-recall curve plot"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (500 Samples)')
        plt.legend(loc="lower left")
        self.save_plot('precision_recall_curve-500')

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

def extract_features_optimized(url):
    """Extract minimal but effective features from a single URL."""
    features = {}
    
    try:
        # Only extract essential URL-based features (fast operations)
        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_equals'] = url.count('=')
        features['num_digits'] = sum(c.isdigit() for c in url)
        
        # Basic domain features
        domain = extract_domain(url)
        if domain:
            features['domain_length'] = len(domain)
            features['is_ip'] = bool(re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', domain))
        
        return features
    except Exception as e:
        logging.error(f"Error extracting features from {url}: {str(e)}")
        return None

def extract_features_batch_optimized(urls, batch_size=50):
    """Extract features from URLs in parallel batches."""
    start_time = time.time()
    all_features = []
    feature_names = None
    
    def process_chunk(url_chunk):
        return [extract_features_optimized(url) for url in url_chunk]
    
    # Process URLs in chunks using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(batch_size, len(urls))) as executor:
        chunks = [urls[i:i+batch_size] for i in range(0, len(urls), batch_size)]
        chunk_futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        
        for future in as_completed(chunk_futures):
            try:
                chunk_features = future.result()
                all_features.extend(chunk_features)
            except Exception as e:
                logging.error(f"Error processing chunk: {str(e)}")
    
    if all_features:
        feature_names = list(all_features[0].keys())
    
    processing_time = time.time() - start_time
    logging.info(f"Batch feature extraction completed in {processing_time:.2f} seconds")
    logging.info(f"Average time per URL: {processing_time/len(urls):.4f} seconds")
    
    return all_features, feature_names

def preprocess_file_data_optimized(phishing_file, legitimate_file, dataset_name, sample_size=None, force_reprocess=False):
    """Optimized preprocessing of URL data."""
    cache_file = f'preprocessed_data/{dataset_name}_cache.pkl'
    
    if not force_reprocess and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Read URLs from CSV files efficiently
    phishing_urls = pd.read_csv(phishing_file, usecols=['url'], nrows=sample_size//2 if sample_size else None)['url'].tolist()
    legitimate_urls = pd.read_csv(legitimate_file, usecols=['url'], nrows=sample_size//2 if sample_size else None)['url'].tolist()
    
    # Process in optimized batches
    start_time = time.time()
    features, feature_names = extract_features_batch_optimized(phishing_urls + legitimate_urls)
    processing_time = time.time() - start_time
    
    # Cache results
    with open(cache_file, 'wb') as f:
        pickle.dump((features, feature_names, processing_time), f)
    
    return features, feature_names, processing_time

def normalize_features(features_list):
    """Normalize features to prevent domination of large-scale features."""
    normalized_features = []
    
    # Get all feature names from first item
    if not features_list:
        return []
    
    feature_names = list(features_list[0].keys())
    
    # Calculate min and max for each feature
    feature_stats = {name: {'min': float('inf'), 'max': float('-inf')} for name in feature_names}
    
    # First pass: find min and max
    for features in features_list:
        for name, value in features.items():
            if isinstance(value, (int, float)):
                feature_stats[name]['min'] = min(feature_stats[name]['min'], value)
                feature_stats[name]['max'] = max(feature_stats[name]['max'], value)
    
    # Second pass: normalize
    for features in features_list:
        normalized = {}
        for name, value in features.items():
            if isinstance(value, (int, float)):
                min_val = feature_stats[name]['min']
                max_val = feature_stats[name]['max']
                if max_val > min_val:
                    normalized[name] = (value - min_val) / (max_val - min_val)
                else:
                    normalized[name] = value
            else:
                normalized[name] = value
        normalized_features.append(normalized)
    
    return normalized_features

def analyze_feature_importance(model, feature_names):
    """Analyze and log feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    logging.info("\nFeature Importance Analysis:")
    for f in range(len(feature_names)):
        logging.info(f"{feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
    
    return dict(zip(feature_names, importances))

def select_features(features_list, importance_dict, threshold=0.01):
    """Select features based on importance threshold."""
    selected_features = [name for name, importance in importance_dict.items() 
                        if importance >= threshold]
    
    logging.info(f"\nSelected {len(selected_features)} features with threshold {threshold}:")
    logging.info(f"Selected features: {selected_features}")
    
    return selected_features

def train_model(X_train, y_train, X_val, y_val, feature_names):
    """Train model with regularization and feature importance analysis."""
    start_time = time.time()
    
    # Initialize model with regularization parameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,  # Prevent overfitting
        min_samples_split=5,  # Minimum samples required to split
        min_samples_leaf=2,   # Minimum samples required at leaf node
        max_features='sqrt',  # Use sqrt of features for each tree
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Analyze feature importance
    importance_dict = analyze_feature_importance(model, feature_names)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    
    training_time = time.time() - start_time
    
    # Log performance metrics
    logging.info(f"\nModel Training Performance:")
    logging.info(f"Training time: {training_time:.2f} seconds")
    logging.info(f"Validation Accuracy: {val_accuracy:.4f}")
    
    return model, importance_dict

def process_and_train(phishing_file, legitimate_file, dataset_name, sample_size=None, force_reprocess=False):
    """Process data and train model with performance logging and feature analysis."""
    logging.info(f"Processing data for dataset: {dataset_name}")
    
    try:
        # Data Processing Phase
        start_processing = time.time()
        logging.info("Starting data processing phase...")
        
        features, feature_names, processing_time = preprocess_file_data_optimized(
            phishing_file, 
            legitimate_file, 
            dataset_name,
            sample_size,
            force_reprocess
        )
        
        # Normalize features
        logging.info("Normalizing features...")
        normalized_features = normalize_features(features)
        
        logging.info(f"Data processing completed in {processing_time:.2f} seconds")
        logging.info(f"Number of features: {len(feature_names)}")
        
        # Data Splitting Phase
        start_splitting = time.time()
        logging.info("Splitting data into train/val/test sets...")
        
        # Convert features to numpy array
        feature_array = np.array([[feat[name] for name in feature_names] 
                                for feat in normalized_features])
        labels = np.array([1] * (len(features)//2) + [0] * (len(features)//2))
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            feature_array, labels
        )
        
        splitting_time = time.time() - start_splitting
        logging.info(f"Data splitting completed in {splitting_time:.2f} seconds")
        
        # Model Training Phase
        start_training = time.time()
        logging.info("Starting model training phase...")
        
        model, importance_dict = train_model(X_train, y_train, X_val, y_val, feature_names)
        
        # Select important features
        selected_features = select_features(normalized_features, importance_dict)
        
        training_time = time.time() - start_training
        total_time = time.time() - start_processing
        
        # Log comprehensive timing information
        logging.info("\nPerformance Summary:")
        logging.info(f"Data Processing Time: {processing_time:.2f} seconds")
        logging.info(f"Data Splitting Time: {splitting_time:.2f} seconds")
        logging.info(f"Model Training Time: {training_time:.2f} seconds")
        logging.info(f"Total Time: {total_time:.2f} seconds")
        logging.info(f"Average time per sample: {total_time/len(features):.4f} seconds")
        
        return model
        
    except Exception as e:
        logging.error(f"Error in process_and_train: {str(e)}")
        raise

def split_data(features, labels, test_size=0.2, val_size=0.2):
    """Split data into training, validation, and test sets."""
    # First split into training and temp sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=(test_size + val_size), random_state=42
    )
    
    # Split temp into validation and test sets
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=42
    )
    
    logging.info(f"Training set size: {len(X_train)}")
    logging.info(f"Validation set size: {len(X_val)}")
    logging.info(f"Test set size: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train, y_train, X_val, y_val):
    """Train the model with the given data."""
    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    logging.info(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    
    return model

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
        features_df = load_and_process_data(phishing_file_path, legitimate_file_path, sample_size=500)
        
        # Split the data
        logging.info("\nSplitting data into train, validation, and test sets...")
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data_splits(features_df)
        
        # Perform hyperparameter tuning with enhanced cross-validation
        best_base_model, best_calibrated_model = tune_random_forest(X_train, y_train)
        
        # Generate learning curve plot
        logging.info("\nGenerating learning curve plot...")
        visualizer.plot_learning_curve(best_base_model, X_train, y_train)
        
        # Evaluate on validation set
        logging.info("\nEvaluating on validation set:")
        y_val_pred, y_val_prob, _ = evaluate_model(best_calibrated_model, X_val, y_val, "Validation Set")
        
        # Generate validation set plots
        visualizer.plot_confusion_matrix(y_val, y_val_pred)
        visualizer.plot_roc_curve(y_val, y_val_prob)
        visualizer.plot_precision_recall_curve(y_val, y_val_prob)
        
        # Final evaluation on test set
        logging.info("\nEvaluating on test set:")
        y_test_pred, y_test_prob, _ = evaluate_model(best_calibrated_model, X_test, y_test, "Test Set")
        
        # Generate test set plots
        logging.info("\nGenerating evaluation plots...")
        visualizer.plot_confusion_matrix(y_test, y_test_pred)
        visualizer.plot_roc_curve(y_test, y_test_prob)
        visualizer.plot_precision_recall_curve(y_test, y_test_prob)
        
        # Feature importance analysis and plot
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': best_base_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info("\nFeature Importance:")
        logging.info(feature_importance)
        visualizer.plot_feature_importance(
            feature_importance['feature'].values,
            feature_importance['importance'].values
        )
        
        # Save the model and feature names
        model_path, feature_names_path = save_model(best_calibrated_model, feature_names)

    finally:
        # Cleanup
        plt.close('all')
        
if __name__ == "__main__":
    main()
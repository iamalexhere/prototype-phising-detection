import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import learning_curve, train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
import logging

# Add parent directory to path to import phishing_detection
sys.path.append(str(Path(__file__).parent.parent))

from phishing_detection import (
    URLFeatureExtractor, 
    load_and_process_data,
    prepare_data_splits,
    extract_domain,
    tune_random_forest,
    ModelVisualizer
)

class TestPhishingDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across multiple tests"""
        cls.test_urls = [
            "https://example.com",
            "http://test.com/path",
            "https://phishing.fake.com",
            "http://legitimate-site.com"
        ]
        
        # Create test CSV files
        cls.test_dir = Path("test_data")
        cls.test_dir.mkdir(exist_ok=True)
        
        # Create phishing URLs file
        cls.phishing_file = cls.test_dir / "test_phishing.csv"
        pd.DataFrame({
            'url': cls.test_urls[:2]
        }).to_csv(cls.phishing_file, index=False)
        
        # Create legitimate URLs file
        cls.legitimate_file = cls.test_dir / "test_legitimate.csv"
        pd.DataFrame({
            'url': cls.test_urls[2:]
        }).to_csv(cls.legitimate_file, index=False)
        
        # Initialize feature extractor
        cls.extractor = URLFeatureExtractor(max_workers=2)

    def test_url_feature_extractor_initialization(self):
        """Test URLFeatureExtractor initialization"""
        self.assertIsNotNone(self.extractor)
        self.assertEqual(self.extractor.max_workers, 2)
        self.assertIsInstance(self.extractor.feature_cache, dict)

    def test_extract_features_single_url(self):
        """Test feature extraction for a single URL"""
        url = "https://example.com"
        features = self.extractor.extract_features(url)
        
        # Check basic feature presence
        self.assertIsInstance(features, dict)
        self.assertTrue('url_length' in features)
        self.assertTrue('domain_length' in features)
        self.assertTrue('is_https' in features)
        
        # Check specific feature values
        self.assertEqual(features['url_length'], len(url))
        self.assertEqual(features['is_https'], 1)

    def test_extract_features_batch(self):
        """Test batch feature extraction"""
        features_list = self.extractor.extract_features_batch(self.test_urls[:2])
        
        # Check basic properties
        self.assertIsInstance(features_list, list)
        self.assertEqual(len(features_list), 2)
        
        # Check each feature dict
        for features in features_list:
            self.assertIsInstance(features, dict)
            self.assertTrue('url_length' in features)
            self.assertTrue('domain_length' in features)

    def test_load_and_process_data(self):
        """Test data loading and processing"""
        features_df = load_and_process_data(
            self.phishing_file,
            self.legitimate_file,
            sample_size=2
        )
        
        # Check DataFrame properties
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), 4)  # 2 phishing + 2 legitimate
        self.assertTrue('url' in features_df.columns)
        self.assertTrue('label' in features_df.columns)
        
        # Check labels
        labels = features_df['label'].unique()
        self.assertTrue(all(label in [0, 1] for label in labels))

    def test_load_and_process_data_small(self):
        """Test data loading and processing with small sample size"""
        features_df = load_and_process_data(
            str(self.test_dir / 'phishing_urls.txt'),
            str(self.test_dir / 'legitimate_urls.txt'),
            sample_size=4
        )
        self.assertIsNotNone(features_df)
        self.assertEqual(len(features_df), 4)

    def test_load_and_process_data_large(self):
        """Test data loading and processing with large sample size handling"""
        # Create larger test datasets
        large_phishing = self.test_dir / 'large_phishing.txt'
        large_legitimate = self.test_dir / 'large_legitimate.txt'
        
        # Generate test URLs
        phishing_urls = [f'http://phishing{i}.fake.com/scam' for i in range(100)]
        legitimate_urls = [f'http://legitimate{i}.com' for i in range(100)]
        
        large_phishing.write_text('\n'.join(phishing_urls))
        large_legitimate.write_text('\n'.join(legitimate_urls))
        
        # Test with larger sample size
        features_df = load_and_process_data(
            str(large_phishing),
            str(large_legitimate),
            sample_size=150
        )
        
        self.assertIsNotNone(features_df)
        self.assertLessEqual(len(features_df), 200)  # Should not exceed available URLs

    def test_prepare_data_splits(self):
        """Test data splitting functionality"""
        # Create a sample DataFrame
        features_df = pd.DataFrame({
            'url': self.test_urls,
            'feature1': np.random.rand(4),
            'feature2': np.random.rand(4),
            'label': [1, 1, 0, 0]
        })
        
        # Test splitting
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data_splits(
            features_df,
            test_size=0.25,
            val_size=0.25
        )
        
        # Check split sizes
        total_samples = len(features_df)
        expected_test_size = int(total_samples * 0.25)
        expected_val_size = int(total_samples * 0.25)
        
        self.assertEqual(len(X_test), expected_test_size)
        self.assertEqual(len(X_val), expected_val_size)
        self.assertEqual(len(y_test), expected_test_size)
        self.assertEqual(len(y_val), expected_val_size)

    def test_extract_domain(self):
        """Test domain extraction function"""
        test_cases = [
            ("https://www.example.com/path", "example.com"),
            ("http://subdomain.test.com", "test.com"),
            ("https://example.co.uk", "example.co.uk"),
            ("invalid-url", None)
        ]
        
        for url, expected in test_cases:
            result = extract_domain(url)
            self.assertEqual(result, expected)

    def test_tune_random_forest(self):
        """Test random forest tuning and calibration"""
        # Create a small synthetic dataset
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Train the model
        best_base_model, best_calibrated_model = tune_random_forest(X, y)
        
        # Check that both models are properly fitted
        self.assertTrue(hasattr(best_base_model, 'classes_'))
        self.assertTrue(hasattr(best_calibrated_model, 'classes_'))
        
        # Test predictions
        X_test = np.random.rand(10, 5)
        base_pred = best_base_model.predict(X_test)
        calibrated_pred = best_calibrated_model.predict(X_test)
        
        self.assertEqual(len(base_pred), 10)
        self.assertEqual(len(calibrated_pred), 10)
        
        # Test probability predictions
        base_prob = best_base_model.predict_proba(X_test)
        calibrated_prob = best_calibrated_model.predict_proba(X_test)
        
        self.assertEqual(base_prob.shape, (10, 2))
        self.assertEqual(calibrated_prob.shape, (10, 2))

    def test_model_training_parameters(self):
        """Test model training with different parameter configurations"""
        # Create synthetic dataset
        X = np.random.rand(1000, 10)  # More features for realistic testing
        y = np.random.randint(0, 2, 1000)
        
        # Test with different parameter configurations
        param_sets = [
            {'n_estimators': 50, 'max_depth': 10},
            {'n_estimators': 100, 'max_depth': None},
            {'n_estimators': 200, 'max_samples': 0.8}
        ]
        
        for params in param_sets:
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X, y)
            
            # Test predictions
            X_test = np.random.rand(100, 10)
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            
            self.assertEqual(len(predictions), 100)
            self.assertEqual(probabilities.shape, (100, 2))

    def test_model_training_pipeline(self):
        """Test the complete model training pipeline"""
        # Create synthetic dataset
        X = np.random.rand(500, 15)  # Similar to actual feature count
        y = np.random.randint(0, 2, 500)
        
        # Split data
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
        
        # Train models
        base_model, calibrated_model = tune_random_forest(X_train, y_train)
        
        # Test base model
        self.assertIsInstance(base_model, RandomForestClassifier)
        self.assertTrue(hasattr(base_model, 'feature_importances_'))
        
        # Test calibrated model
        self.assertIsInstance(calibrated_model, CalibratedClassifierCV)
        
        # Test predictions
        test_pred = calibrated_model.predict(X_test)
        test_prob = calibrated_model.predict_proba(X_test)
        
        self.assertEqual(len(test_pred), len(y_test))
        self.assertEqual(test_prob.shape[0], len(y_test))
        self.assertEqual(test_prob.shape[1], 2)

    def test_learning_curve_generation(self):
        """Test learning curve generation with different dataset sizes"""
        visualizer = ModelVisualizer(output_dir=str(self.test_dir / 'test_plots'))
        
        # Test with different dataset sizes
        sizes = [100, 500]
        for size in sizes:
            X = np.random.rand(size, 10)
            y = np.random.randint(0, 2, size)
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Generate learning curve
            visualizer.plot_learning_curve(model, X, y)
            
            # Check if plot was created with correct name
            plot_file = self.test_dir / 'test_plots' / f'learning_curve-{size}.png'
            self.assertTrue(plot_file.exists())

    def test_model_visualizer(self):
        """Test model visualization functions"""
        # Create temporary directory for plots
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = ModelVisualizer(output_dir=tmpdir)
            
            # Create synthetic data
            X = np.random.rand(100, 5)
            y = np.random.randint(0, 2, 100)
            
            # Create and fit a simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Test learning curve plot
            visualizer.plot_learning_curve(model, X, y)
            
            # Check if plot was created
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'learning_curve.png')))

    def test_model_scalability(self):
        """Test model's ability to handle larger datasets"""
        # Create larger synthetic dataset
        X = np.random.rand(5000, 15)
        y = np.random.randint(0, 2, 5000)
        
        # Split data
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
        
        # Time the training process
        start_time = time.time()
        base_model, calibrated_model = tune_random_forest(X_train, y_train)
        training_time = time.time() - start_time
        
        # Check training time is reasonable (adjust threshold as needed)
        self.assertLess(training_time, 300)  # Should complete within 5 minutes
        
        # Test prediction speed
        start_time = time.time()
        predictions = calibrated_model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Check prediction time is reasonable
        self.assertLess(prediction_time, 5)  # Should predict within 5 seconds
        
        # Verify accuracy is reasonable
        accuracy = accuracy_score(y_test, predictions)
        self.assertGreater(accuracy, 0.5)  # Should be better than random guessing

    def test_performance_100_samples(self):
        """Test performance with exactly 100 samples for both processing and training"""
        # Generate 100 test URLs (50 each for phishing and legitimate) with different domains
        phishing_urls = [f'http://test-phishing{i}.com/page?id={i}' for i in range(50)]
        legitimate_urls = [f'http://test-legitimate{i}.org/home?user={i}' for i in range(50)]
        
        # Create test files
        phishing_path = self.test_dir / 'perf_test_phishing_100.txt'
        legitimate_path = self.test_dir / 'perf_test_legitimate_100.txt'
        
        phishing_path.write_text('\n'.join(phishing_urls))
        legitimate_path.write_text('\n'.join(legitimate_urls))
        
        # Test data processing performance
        processing_start = time.time()
        features_df = load_and_process_data(
            str(phishing_path),
            str(legitimate_path),
            sample_size=100,
            batch_size=25  # Process in smaller batches
        )
        processing_time = time.time() - processing_start
        
        # Log data processing metrics
        logging.info("\nData Processing Performance (100 samples):")
        logging.info(f"Total processing time: {processing_time:.2f} seconds")
        logging.info(f"Average time per URL: {processing_time/100:.4f} seconds")
        logging.info(f"Number of features: {len(features_df.columns) - 2}")  # Excluding label and url columns
        
        # Prepare data for model training
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data_splits(features_df)
        
        # Test model training performance
        training_start = time.time()
        base_model, calibrated_model = tune_random_forest(X_train, y_train)
        training_time = time.time() - training_start
        
        # Make predictions for timing
        prediction_start = time.time()
        y_train_pred = calibrated_model.predict(X_train)  # Use training set for prediction test
        prediction_time = time.time() - prediction_start
        
        # Calculate model performance metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Log model training metrics
        logging.info("\nModel Training Performance (100 samples):")
        logging.info(f"Total training time: {training_time:.2f} seconds")
        logging.info(f"Prediction time: {prediction_time:.4f} seconds")
        logging.info(f"Training accuracy: {train_accuracy:.4f}")
        
        # Performance assertions
        self.assertLess(processing_time, 30)  # Should process 100 URLs within 30 seconds
        self.assertLess(training_time, 60)    # Should train model within 60 seconds
        self.assertLess(prediction_time, 1)   # Should predict within 1 second
        self.assertGreater(train_accuracy, 0.5) # Should be better than random guessing
        
        # Data quality assertions
        self.assertGreaterEqual(len(features_df), 90)  # Allow for some failed feature extractions
        self.assertLess(len(features_df), 101)  # But shouldn't have more than input
        self.assertGreater(len(feature_names), 0)
        self.assertTrue(all(col in features_df.columns for col in ['url', 'label']))

    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        if cls.test_dir.exists():
            for file in cls.test_dir.glob("*"):
                file.unlink()
            cls.test_dir.rmdir()

if __name__ == '__main__':
    unittest.main(verbosity=2)

import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

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

    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        if cls.test_dir.exists():
            for file in cls.test_dir.glob("*"):
                file.unlink()
            cls.test_dir.rmdir()

if __name__ == '__main__':
    unittest.main(verbosity=2)

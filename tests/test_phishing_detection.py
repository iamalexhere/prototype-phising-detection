import unittest
import pandas as pd
import numpy as np
import os
from pathlib import Path
import tempfile
import time
import logging
import sys

# Add parent directory to path to import phishing_detection
sys.path.append(str(Path(__file__).parent.parent))

from phishing_detection import (
    process_and_train,
    extract_features_optimized,
    extract_features_batch_optimized
)

class TestPhishingDetectionPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and directories"""
        cls.test_dir = Path("test_data")
        cls.test_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Generate test URLs for different sample sizes
        cls.sample_sizes = [50, 100]
        cls.test_files = {}
        
        for size in cls.sample_sizes:
            # Generate unique URLs for each size
            phishing_urls = [f'http://test-phishing{i}-{size}.com/page?id={i}' for i in range(size//2)]
            legitimate_urls = [f'http://test-legitimate{i}-{size}.org/home?user={i}' for i in range(size//2)]
            
            # Create test files
            phishing_path = cls.test_dir / f'perf_test_phishing_{size}.csv'
            legitimate_path = cls.test_dir / f'perf_test_legitimate_{size}.csv'
            
            # Save URLs to CSV files
            pd.DataFrame({'url': phishing_urls}).to_csv(phishing_path, index=False)
            pd.DataFrame({'url': legitimate_urls}).to_csv(legitimate_path, index=False)
            
            cls.test_files[size] = {
                'phishing': str(phishing_path),
                'legitimate': str(legitimate_path)
            }

    def test_feature_extraction_performance(self):
        """Test the performance of feature extraction for different sample sizes"""
        for size in self.sample_sizes:
            # Test single URL feature extraction
            test_url = f'http://test-performance.com/page?id={size}'
            
            start_time = time.time()
            features = extract_features_optimized(test_url)
            single_extraction_time = time.time() - start_time
            
            logging.info(f"\nSingle URL Feature Extraction (Sample Size {size}):")
            logging.info(f"Time taken: {single_extraction_time:.4f} seconds")
            
            # Verify features
            self.assertIsNotNone(features)
            self.assertIsInstance(features, dict)
            self.assertGreater(len(features), 0)
            
            # Test batch feature extraction
            test_urls = [f'http://test-batch{i}.com/page?id={i}' for i in range(size)]
            
            start_time = time.time()
            features_batch, feature_names = extract_features_batch_optimized(test_urls)
            batch_extraction_time = time.time() - start_time
            
            logging.info(f"\nBatch Feature Extraction (Sample Size {size}):")
            logging.info(f"Total time: {batch_extraction_time:.2f} seconds")
            logging.info(f"Average time per URL: {batch_extraction_time/size:.4f} seconds")
            
            # Performance assertions
            self.assertLess(single_extraction_time, 1.0)  # Single URL should be processed within 1 second
            self.assertLess(batch_extraction_time/size, 0.1)  # Each URL in batch should take less than 0.1 seconds

    def test_end_to_end_performance(self):
        """Test end-to-end performance including data processing and model training"""
        for size in self.sample_sizes:
            logging.info(f"\nTesting end-to-end performance with {size} samples")
            
            start_time = time.time()
            
            try:
                # Process and train model
                model = process_and_train(
                    self.test_files[size]['phishing'],
                    self.test_files[size]['legitimate'],
                    f'test_dataset_{size}',
                    sample_size=size,
                    force_reprocess=True  # Force reprocessing to measure actual performance
                )
                
                total_time = time.time() - start_time
                
                logging.info(f"\nEnd-to-End Performance Metrics (Sample Size {size}):")
                logging.info(f"Total processing and training time: {total_time:.2f} seconds")
                logging.info(f"Average time per sample: {total_time/size:.4f} seconds")
                
                # Performance assertions
                if size == 50:
                    self.assertLess(total_time, 15)  # Should process 50 samples within 15 seconds
                elif size == 100:
                    self.assertLess(total_time, 25)  # Should process 100 samples within 25 seconds
                
                # Verify model
                self.assertIsNotNone(model)
                
            except Exception as e:
                self.fail(f"Failed processing {size} samples: {str(e)}")

    def test_cached_performance(self):
        """Test performance with cached data"""
        for size in self.sample_sizes:
            logging.info(f"\nTesting cached performance with {size} samples")
            
            # First run to cache the data
            _ = process_and_train(
                self.test_files[size]['phishing'],
                self.test_files[size]['legitimate'],
                f'test_dataset_{size}',
                sample_size=size
            )
            
            # Second run with cached data
            start_time = time.time()
            model = process_and_train(
                self.test_files[size]['phishing'],
                self.test_files[size]['legitimate'],
                f'test_dataset_{size}',
                sample_size=size
            )
            cached_time = time.time() - start_time
            
            logging.info(f"\nCached Performance Metrics (Sample Size {size}):")
            logging.info(f"Total processing time with cache: {cached_time:.2f} seconds")
            logging.info(f"Average time per sample: {cached_time/size:.4f} seconds")
            
            # Performance assertions for cached data
            self.assertLess(cached_time, 5)  # Cached processing should be very fast
            self.assertIsNotNone(model)

    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        if cls.test_dir.exists():
            for file in cls.test_dir.glob("*"):
                try:
                    file.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete {file}: {str(e)}")
            try:
                cls.test_dir.rmdir()
            except Exception as e:
                logging.warning(f"Failed to delete test directory: {str(e)}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
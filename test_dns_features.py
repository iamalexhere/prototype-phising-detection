import unittest
from dns_features import extract_dns_features
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestDNSFeatures(unittest.TestCase):
    def setUp(self):
        self.test_url = "https://studentportal.unpar.ac.id"
        
    def test_dns_feature_extraction(self):
        """Test DNS feature extraction for studentportal.unpar.ac.id"""
        result = extract_dns_features(self.test_url)
        
        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check if result is not empty
        self.assertFalse(result.empty)
        
        # Get the features as a dictionary
        features = result.iloc[0].to_dict()
        
        # Basic URL checks
        self.assertEqual(features['url'], self.test_url)
        
        # DNS record checks
        self.assertIn('has_a_record', features)
        self.assertIn('num_a_records', features)
        self.assertIn('has_mx_record', features)
        self.assertIn('num_mx_records', features)
        self.assertIn('has_ns_record', features)
        self.assertIn('num_ns_records', features)
        
        # WHOIS checks
        self.assertIn('domain_age_days', features)
        self.assertIn('is_domain_young', features)
        self.assertIn('days_to_expiration', features)
        self.assertIn('is_expiring_soon', features)
        self.assertIn('has_registrar', features)
        self.assertIn('has_registrant', features)
        
        # SSL checks
        self.assertIn('ssl_days_valid', features)
        self.assertIn('ssl_is_valid', features)
        self.assertIn('ssl_is_expired', features)
        self.assertIn('ssl_is_self_signed', features)
        
        # Log the results
        logger.info("DNS Features Test Results:")
        for key, value in features.items():
            logger.info(f"{key}: {value}")
            
    def test_ip_feature_extraction(self):
        """Test IP-related features"""
        result = extract_dns_features(self.test_url)
        features = result.iloc[0].to_dict()
        
        # Check IP-related features
        self.assertIn('ip_address', features)
        self.assertIn('is_private_ip', features)
        
        if features['has_a_record']:
            self.assertIsNotNone(features['ip_address'])
            self.assertIsInstance(features['is_private_ip'], int)
            logger.info(f"IP Address: {features['ip_address']}")
            logger.info(f"Is Private IP: {bool(features['is_private_ip'])}")
            
    def test_ssl_feature_extraction(self):
        """Test SSL certificate features"""
        result = extract_dns_features(self.test_url)
        features = result.iloc[0].to_dict()
        
        # SSL validity checks
        self.assertIsInstance(features['ssl_days_valid'], (int, float))
        self.assertIsInstance(features['ssl_is_valid'], int)
        self.assertIsInstance(features['ssl_is_expired'], int)
        self.assertIsInstance(features['ssl_is_self_signed'], int)
        
        logger.info("SSL Features:")
        logger.info(f"Days Valid: {features['ssl_days_valid']}")
        logger.info(f"Is Valid: {bool(features['ssl_is_valid'])}")
        logger.info(f"Is Expired: {bool(features['ssl_is_expired'])}")
        logger.info(f"Is Self-Signed: {bool(features['ssl_is_self_signed'])}")

if __name__ == '__main__':
    unittest.main(verbosity=2)

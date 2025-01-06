from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from phishing_detection import URLFeatureExtractor
import logging
from pathlib import Path
import json
from datetime import datetime
from screenshot import capture_website_screenshot
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import io
import whois
import socket
import requests
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor
import dns.resolver
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_latest_model(models_dir='models/split_10'):
    """Load the latest trained model and its feature names"""
    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory {models_dir} not found")
    
    model_files = list(models_path.glob("phishing_detector_split_9_*.joblib"))
    feature_files = list(models_path.glob("feature_names_split_9_*.joblib"))
    
    if not model_files or not feature_files:
        raise FileNotFoundError("Model or feature names file not found")
    
    latest_model = max(model_files, key=lambda x: x.name)
    latest_features = max(feature_files, key=lambda x: x.name)
    
    model = joblib.load(latest_model)
    feature_names = joblib.load(latest_features)
    
    return model, feature_names, latest_model.name

def normalize_feature_vector(feature_vector, feature_names):
    """Normalize features to a common scale"""
    normalized = feature_vector.copy()
    
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

def get_risk_classification(probability):
    """
    Get detailed risk classification based on final adjusted probability score
    Returns dictionary with classification details
    """
    if probability < 0.2:
        return {
            "level": "Safe",
            "description": "This website has passed our security checks",
            "color": "success",
            "icon": "fa-check-circle",
            "details": "After analyzing all security factors including domain history, SSL certificate, and DNS records, this website shows strong signs of being legitimate"
        }
    elif probability < 0.4:
        return {
            "level": "Low Risk",
            "description": "This website appears mostly safe but has minor security concerns",
            "color": "info",
            "icon": "fa-info-circle",
            "details": "While the overall security score is good, there are some recommended security improvements that could be made"
        }
    elif probability < 0.6:
        return {
            "level": "Medium Risk",
            "description": "This website has some concerning security issues",
            "color": "warning",
            "icon": "fa-exclamation-circle",
            "details": "Our analysis found several security concerns. While not definitively malicious, exercise caution and verify the website's legitimacy before proceeding"
        }
    elif probability < 0.8:
        return {
            "level": "High Risk",
            "description": "Multiple security issues detected - likely malicious",
            "color": "danger",
            "icon": "fa-exclamation-triangle",
            "details": "This website exhibits many characteristics commonly associated with phishing attempts. It is strongly recommended to avoid entering any sensitive information"
        }
    else:
        return {
            "level": "Critical Risk",
            "description": "This website is almost certainly malicious",
            "color": "critical",
            "icon": "fa-skull-crossbones",
            "details": "Our analysis indicates this is very likely a phishing website. DO NOT proceed or enter any information. If you've already shared any data, consider it compromised"
        }

def analyze_url(url):
    """Analyze URL and return phishing prediction results"""
    try:
        # Load model
        model, feature_names, model_name = load_latest_model()
        logger.info(f"Using model: {model_name}")
        logger.info(f"Loaded {len(feature_names)} features: {', '.join(feature_names)}")
        
        # Extract features
        extractor = URLFeatureExtractor()
        features = extractor.extract_features(url)
        
        if features is None:
            return {"error": "Failed to extract features from URL"}
            
        # Add is_private_ip feature
        if features.get('has_ip', 0):
            import re
            ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
            ip_match = re.search(ip_pattern, url)
            if ip_match:
                features['is_private_ip'] = 1 if is_private_ip(ip_match.group(0)) else 0
            else:
                features['is_private_ip'] = 0
        else:
            features['is_private_ip'] = 0
            
        # Prepare feature vector
        feature_dict = {name: 0 for name in feature_names}
        feature_dict.update({k: 1 if isinstance(v, bool) and v else 0 if isinstance(v, bool) else v 
                           for k, v in features.items()})
        feature_vector = pd.DataFrame([feature_dict])[feature_names]
        
        # Normalize features
        feature_vector = normalize_feature_vector(feature_vector, feature_names)
        
        # Get prediction probability
        probability = model.predict_proba(feature_vector)[0]
        phishing_prob = probability[1] * 100  # Convert to percentage
        
        # Try to capture screenshot
        screenshot_success, screenshot_path = capture_website_screenshot(url)
        
        # Calculate trust score
        trust_score = 0.0
        
        # DNS Group (46.01% importance)
        if features.get('has_mx_record', 0):
            trust_score += 0.20  # Most important feature (20.22%)
        if features.get('num_mx_records', 0) > 1:
            trust_score += 0.18  # Second most important (18.45%)
        if features.get('has_a_record', 0):
            trust_score += 0.06  # 5.58% importance
        normalized_domain_age = feature_vector['domain_age_days'].iloc[0] if 'domain_age_days' in feature_names else 0
        if normalized_domain_age > 0.7:
            trust_score += 0.04  # 4.20% importance
            
        # SSL Group (26.14% importance)
        ssl_days_valid = feature_vector['ssl_days_valid'].iloc[0] if 'ssl_days_valid' in feature_names else 0
        if ssl_days_valid > 0.8:
            trust_score += 0.14  # 14.31% importance
        if features.get('ssl_is_valid', 0):
            trust_score += 0.12  # 11.83% importance
            
        # URL Group (16.66% importance)
        url_length = feature_vector['url_length'].iloc[0] if 'url_length' in feature_names else 0
        if 0.3 <= url_length <= 0.7:  # Moderate URL length
            trust_score += 0.09  # 8.70% importance
        if not features.get('has_multiple_subdomains', 0):
            trust_score += 0.05  # 4.52% importance
        if not features.get('suspicious_tld', 0):
            trust_score += 0.01  # 1.29% importance
            
        # Prepare insights
        insights = {
            'domain_health': {
                'title': 'Domain Health',
                'status': 'Good' if features.get('domain_age_days', 0) > 180 else 'Suspicious',
                'details': []
            },
            'dns_security': {
                'title': 'DNS Security',
                'status': 'Secure' if all([
                    features.get('has_mx_record', 0),
                    features.get('has_ns_record', 0),
                    features.get('num_ns_records', 0) >= 2
                ]) else 'Incomplete',
                'details': []
            },
            'ssl_status': {
                'title': 'SSL Security',
                'status': 'Secure' if all([
                    features.get('ssl_is_valid', 0),
                    features.get('ssl_days_valid', 0) > 90
                ]) else 'Insecure',
                'details': []
            }
        }
        
        # Domain Health Details
        domain_age = features.get('domain_age_days', 0)
        if domain_age > 0:
            age_years = domain_age / 365
            insights['domain_health']['details'].append(
                f"Domain age: {age_years:.1f} years" if age_years >= 1 else f"Domain age: {int(domain_age)} days"
            )
        if features.get('has_registrar'):
            insights['domain_health']['details'].append("Domain has valid registrar information")
        if features.get('days_to_expiration'):
            days_left = features.get('days_to_expiration')
            insights['domain_health']['details'].append(
                f"Domain expires in {int(days_left)} days"
            )
            
        # DNS Security Details
        if features.get('has_mx_record'):
            mx_count = features.get('num_mx_records', 0)
            insights['dns_security']['details'].append(
                f"Has {mx_count} mail server{'s' if mx_count > 1 else ''}"
            )
        if features.get('has_ns_record'):
            ns_count = features.get('num_ns_records', 0)
            insights['dns_security']['details'].append(
                f"Has {ns_count} name server{'s' if ns_count > 1 else ''}"
            )
        if features.get('has_a_record'):
            insights['dns_security']['details'].append("Domain resolves to valid IP")
            if features.get('is_private_ip'):
                insights['dns_security']['details'].append("WARNING: Resolves to private IP address")
        
        # SSL Security Details
        if features.get('ssl_is_valid', 0):
            insights['ssl_status']['details'].append("Valid SSL certificate")
            if features.get('ssl_days_valid', 0) > 365:
                insights['ssl_status']['details'].append("Long-term SSL certificate")
        else:
            insights['ssl_status']['details'].append("WARNING: Invalid or missing SSL certificate")
        
        # Calculate adjusted probability
        adjusted_prob = max(0.01, min(0.99, phishing_prob / 100 - trust_score))
        is_phishing = adjusted_prob > 0.45
        
        # Get risk classification based on final score
        risk_class = get_risk_classification(adjusted_prob)
        
        # Prepare trust indicators
        trust_indicators = []
        warning_indicators = []
        
        # SSL indicators
        if features.get('ssl_is_valid', 0):
            trust_indicators.append(f"Valid SSL certificate (expires in {int(features.get('ssl_days_valid', 0))} days)")
            if features.get('ssl_days_valid', 0) > 365:
                trust_indicators.append("Long-term SSL certificate")
        else:
            warning_indicators.append("Invalid or missing SSL certificate")
            
        # Domain age indicators
        domain_age = features.get('domain_age_days', 0)
        if domain_age > 365:
            trust_indicators.append(f"Domain is {domain_age/365:.1f} years old")
        elif domain_age > 180:
            trust_indicators.append(f"Domain is {int(domain_age)} days old")
        else:
            warning_indicators.append(f"Domain is only {int(domain_age)} days old")
            
        # DNS record indicators
        if features.get('has_mx_record', 0):
            trust_indicators.append(f"Has {features.get('num_mx_records', 0)} mail servers")
        if features.get('has_ns_record', 0):
            trust_indicators.append(f"Has {features.get('num_ns_records', 0)} name servers")
            
        # URL structure indicators
        if features.get('has_ip', 0):
            warning_indicators.append("Uses IP address instead of domain name")
        if features.get('has_at_symbol', 0):
            warning_indicators.append("Contains @ symbol in URL")
        if features.get('has_multiple_subdomains', 0):
            warning_indicators.append("Uses multiple subdomains")
        if features.get('has_double_slash', 0):
            warning_indicators.append("Contains double slash in path")
            
        # Extract domain for additional analysis
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Gather enhanced security metrics
        whois_info = get_whois_info(domain)
        reputation_data = check_reputation(domain)
        hosting_info = analyze_hosting(domain)
        redirect_chain = check_redirect_chain(url)
        traceroute_info = perform_traceroute(domain)
        
        # Add network analysis results to insights
        insights['network_analysis'] = {
            'title': 'Network Analysis',
            'status': 'Secure' if reputation_data['reputation_score'] >= 70 else 'Suspicious',
            'details': [
                f"Reputation Score: {reputation_data['reputation_score']}/100",
                f"Hosting Provider: {hosting_info.get('hosting_provider', 'Unknown')}",
                f"Server Location: {hosting_info.get('country', 'Unknown')}",
                f"Redirect Chain Length: {len(redirect_chain)}",
                f"Network Hops: {len(traceroute_info)}"
            ]
        }
        
        insights['domain_registration'] = {
            'title': 'Domain Registration',
            'status': 'Verified' if whois_info else 'Unknown',
            'details': [
                f"Registrar: {whois_info.get('registrar', 'Unknown')}",
                f"Registration Date: {whois_info.get('creation_date', 'Unknown')}",
                f"Organization: {whois_info.get('registrant_org', 'Unknown')}"
            ]
        }
        
        # Add detailed analysis results
        analysis_results = {
            "url": url,
            "raw_probability": float(phishing_prob),
            "trust_score": float(trust_score * 100),
            "adjusted_probability": float(adjusted_prob * 100),
            "is_phishing": bool(is_phishing),
            "risk_classification": risk_class,
            "trust_indicators": list(trust_indicators),
            "warning_indicators": list(warning_indicators),
            "features": {k: str(float(v)) if isinstance(v, (int, float)) else str(v) 
                       for k, v in features.items()},
            "model_name": str(model_name),
            "insights": dict(insights),
            "screenshot": str(screenshot_path) if screenshot_success else None,
            "network_details": {
                'whois_info': whois_info,
                'reputation_data': reputation_data,
                'hosting_info': hosting_info,
                'redirect_chain': redirect_chain,
                'traceroute_info': traceroute_info
            }
        }
        
        # Adjust trust score based on new metrics
        if reputation_data['reputation_score'] >= 70:
            trust_score += 0.15
        if not reputation_data['blacklisted']:
            trust_score += 0.10
        if hosting_info.get('is_hosting', False):
            trust_score += 0.05
        if len(redirect_chain) <= 2:  # Fewer redirects is better
            trust_score += 0.05
            
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error analyzing URL: {str(e)}")
        return {"error": str(e)}

def is_private_ip(ip):
    """Check if an IP address is private"""
    try:
        import ipaddress
        return ipaddress.ip_address(ip).is_private
    except:
        return False

def process_qr_code(file_storage):
    """Process QR code image and extract URL"""
    try:
        # Read image file into memory
        in_memory_file = io.BytesIO()
        file_storage.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        
        # Decode image
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode image")
            
        # Detect and decode QR code
        decoded_objects = decode(img)
        if not decoded_objects:
            raise ValueError("No QR code found in image")
            
        # Get URL from QR code
        qr_data = decoded_objects[0].data.decode('utf-8')
        if not qr_data.startswith(('http://', 'https://')):
            raise ValueError("QR code does not contain a valid URL")
            
        return qr_data
        
    except Exception as e:
        logger.error(f"Error processing QR code: {str(e)}")
        raise ValueError(f"Failed to process QR code: {str(e)}")

def get_whois_info(domain: str) -> Dict[str, Any]:
    """Get detailed WHOIS information for a domain"""
    try:
        w = whois.whois(domain)
        return {
            'registrar': w.registrar,
            'creation_date': str(w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date),
            'expiration_date': str(w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date),
            'registrant_country': w.registrant_country,
            'registrant_org': w.org,
            'last_updated': str(w.updated_date[0] if isinstance(w.updated_date, list) else w.updated_date)
        }
    except Exception as e:
        logger.error(f"WHOIS lookup failed: {str(e)}")
        return {}

def check_reputation(domain: str) -> Dict[str, Any]:
    """Check domain reputation from multiple sources"""
    reputation_data = {
        'blacklisted': False,
        'reputation_score': 0,
        'threat_categories': [],
        'sources_checked': []
    }
    
    # Check against common blacklists
    blacklists = [
        'zen.spamhaus.org',
        'bl.spamcop.net',
        'dnsbl.sorbs.net'
    ]
    
    def check_blacklist(bl):
        try:
            addr = f"{domain}.{bl}"
            dns.resolver.resolve(addr, 'A')
            return True
        except:
            return False
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(check_blacklist, blacklists))
        
    reputation_data['blacklisted'] = any(results)
    reputation_data['sources_checked'] = blacklists
    
    # Calculate reputation score (0-100)
    reputation_data['reputation_score'] = 100 - (sum(results) * 33)
    
    return reputation_data

def perform_traceroute(domain: str) -> List[Dict[str, str]]:
    """Perform traceroute analysis"""
    try:
        # Using tracert for Windows
        process = subprocess.Popen(['tracert', '-h', '15', domain],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        output, _ = process.communicate()
        
        hops = []
        for line in output.split('\n'):
            # Skip empty lines and header lines
            if not line.strip() or 'Tracing route to' in line or 'over a maximum' in line:
                continue
                
            # Parse only lines that contain timing information
            if 'ms' in line:
                parts = line.strip().split()
                
                # Find hop number
                hop_number = next((p for p in parts if p.isdigit()), '0')
                
                # Find IP address or hostname (usually the last part)
                ip = parts[-1] if parts else 'Unknown'
                
                # Find the best response time
                response_time = 'Unknown'
                for part in parts:
                    if 'ms' in part:
                        response_time = part
                        break
                
                hop = {
                    'hop_number': hop_number,
                    'ip': ip,
                    'response_time': response_time
                }
                hops.append(hop)
                
        return hops
    except Exception as e:
        logger.error(f"Traceroute failed: {str(e)}")
        return []

def analyze_hosting(domain: str) -> Dict[str, Any]:
    """Analyze hosting provider information"""
    try:
        ip = socket.gethostbyname(domain)
        
        # Get hosting provider info using ip-api.com (free API)
        response = requests.get(f'http://ip-api.com/json/{ip}')
        if response.status_code == 200:
            data = response.json()
            return {
                'ip': ip,
                'hosting_provider': data.get('isp', 'Unknown'),
                'organization': data.get('org', 'Unknown'),
                'country': data.get('country', 'Unknown'),
                'city': data.get('city', 'Unknown'),
                'is_hosting': data.get('hosting', False)
            }
    except Exception as e:
        logger.error(f"Hosting analysis failed: {str(e)}")
    return {}

def check_redirect_chain(url: str, max_redirects: int = 5) -> List[Dict[str, str]]:
    """Monitor URL redirect chain"""
    redirects = []
    try:
        response = requests.get(url, allow_redirects=True)
        for resp in response.history:
            redirects.append({
                'url': resp.url,
                'status_code': resp.status_code
            })
        # Add final destination
        redirects.append({
            'url': response.url,
            'status_code': response.status_code
        })
    except Exception as e:
        logger.error(f"Redirect chain analysis failed: {str(e)}")
    return redirects

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if the request has form data (file upload)
        if 'qr_image' in request.files:
            file = request.files['qr_image']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
                
            if not file.content_type.startswith('image/'):
                return jsonify({"error": "File must be an image"}), 400
                
            try:
                # Process QR code and get URL
                url = process_qr_code(file)
                
                # Analyze the extracted URL
                result = analyze_url(url)
                result['url'] = url  # Include the extracted URL in response
                return jsonify(result)
                
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            
        # Handle JSON data for direct URL analysis
        elif request.is_json:
            data = request.get_json()
            url = data.get('url', '').strip()
            
            if not url:
                return jsonify({"error": "URL is required"}), 400
                
            result = analyze_url(url)
            return jsonify(result)
            
        else:
            return jsonify({"error": "Invalid request format"}), 400
            
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

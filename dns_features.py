import pandas as pd
import dns.resolver
import whois
import socket
from datetime import datetime
import ssl
import OpenSSL.SSL
from urllib.parse import urlparse
import tldextract
import requests

def extract_dns_features(url):
    """
    Extract DNS-related features from a URL
    
    Features:
    - DNS record existence (A, MX, NS records)
    - Domain age and expiration
    - SSL certificate validity
    - IP blacklist status
    - WHOIS registration details
    """
    try:
        # Initialize feature dictionary
        features = {'url': url}
        
        # Parse URL
        parsed = urlparse(url)
        extracted = tldextract.extract(url)
        domain = parsed.netloc if parsed.netloc else url
        
        # Remove www. if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        # DNS Records Check
        resolver = dns.resolver.Resolver()
        resolver.timeout = 2
        resolver.lifetime = 2
        
        # Check A record
        try:
            a_records = resolver.resolve(domain, 'A')
            features['has_a_record'] = 1
            features['num_a_records'] = len(a_records)
            
            # Get IP address
            ip = str(a_records[0])
            features['ip_address'] = ip
            
            # Check if IP is private
            try:
                ip_parts = list(map(int, ip.split('.')))
                is_private = (
                    (ip_parts[0] == 10) or
                    (ip_parts[0] == 172 and ip_parts[1] >= 16 and ip_parts[1] <= 31) or
                    (ip_parts[0] == 192 and ip_parts[1] == 168)
                )
                features['is_private_ip'] = int(is_private)
            except:
                features['is_private_ip'] = 0
                
        except:
            features['has_a_record'] = 0
            features['num_a_records'] = 0
            features['ip_address'] = None
            features['is_private_ip'] = 0
            
        # Check MX record
        try:
            mx_records = resolver.resolve(domain, 'MX')
            features['has_mx_record'] = 1
            features['num_mx_records'] = len(mx_records)
        except:
            features['has_mx_record'] = 0
            features['num_mx_records'] = 0
            
        # Check NS record
        try:
            ns_records = resolver.resolve(domain, 'NS')
            features['has_ns_record'] = 1
            features['num_ns_records'] = len(ns_records)
        except:
            features['has_ns_record'] = 0
            features['num_ns_records'] = 0
            
        # WHOIS Information
        try:
            whois_info = whois.whois(domain)
            
            # Creation date
            creation_date = whois_info.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
                
            if creation_date:
                domain_age = (datetime.now() - creation_date).days
                features['domain_age_days'] = domain_age
                features['is_domain_young'] = int(domain_age < 365)  # Domain less than 1 year old
            else:
                features['domain_age_days'] = -1
                features['is_domain_young'] = 1
                
            # Expiration date
            expiration_date = whois_info.expiration_date
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]
                
            if expiration_date:
                days_to_expiration = (expiration_date - datetime.now()).days
                features['days_to_expiration'] = days_to_expiration
                features['is_expiring_soon'] = int(days_to_expiration < 30)  # Expiring within 30 days
            else:
                features['days_to_expiration'] = -1
                features['is_expiring_soon'] = 1
                
            # Registration details
            features['has_registrar'] = int(bool(whois_info.registrar))
            features['has_registrant'] = int(bool(whois_info.registrant))
            
        except Exception as e:
            features['domain_age_days'] = -1
            features['is_domain_young'] = 1
            features['days_to_expiration'] = -1
            features['is_expiring_soon'] = 1
            features['has_registrar'] = 0
            features['has_registrant'] = 0
            
        # SSL Certificate Check
        if parsed.scheme == 'https':
            try:
                cert = ssl.get_server_certificate((domain, 443))
                x509 = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, cert)
                
                # Check validity
                not_after = datetime.strptime(x509.get_notAfter().decode('ascii'), '%Y%m%d%H%M%SZ')
                not_before = datetime.strptime(x509.get_notBefore().decode('ascii'), '%Y%m%d%H%M%SZ')
                
                features['ssl_days_valid'] = (not_after - datetime.now()).days
                features['ssl_is_valid'] = int(datetime.now() > not_before and datetime.now() < not_after)
                features['ssl_is_expired'] = int(datetime.now() > not_after)
                
                # Check if self-signed
                features['ssl_is_self_signed'] = int(x509.get_subject() == x509.get_issuer())
                
            except Exception as e:
                features['ssl_days_valid'] = -1
                features['ssl_is_valid'] = 0
                features['ssl_is_expired'] = 1
                features['ssl_is_self_signed'] = 1
        else:
            features['ssl_days_valid'] = -1
            features['ssl_is_valid'] = 0
            features['ssl_is_expired'] = 1
            features['ssl_is_self_signed'] = 1
            
        # Convert to DataFrame
        return pd.DataFrame([features])
        
    except Exception as e:
        print(f"Error extracting DNS features for URL {url}: {str(e)}")
        return None

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
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_dns_info(domain):
    """Alternative approach to get DNS information using socket and requests"""
    features = {}
    try:
        # Try to resolve A record using socket
        logger.debug(f"Attempting to resolve {domain} using socket")
        ip_address = socket.gethostbyname(domain)
        features['has_a_record'] = 1
        features['num_a_records'] = 1
        features['ip_address'] = ip_address
        logger.debug(f"Successfully resolved {domain} to {ip_address}")
        
        # Check if IP is private
        ip_parts = list(map(int, ip_address.split('.')))
        is_private = (
            (ip_parts[0] == 10) or
            (ip_parts[0] == 172 and ip_parts[1] >= 16 and ip_parts[1] <= 31) or
            (ip_parts[0] == 192 and ip_parts[1] == 168)
        )
        features['is_private_ip'] = int(is_private)
    except Exception as e:
        logger.error(f"Failed to resolve {domain} using socket: {str(e)}")
        features['has_a_record'] = 0
        features['num_a_records'] = 0
        features['ip_address'] = None
        features['is_private_ip'] = 0
    
    # Try to get MX and NS records using requests to external DNS API
    try:
        logger.debug(f"Querying DNS records using external API")
        # Get MX records for the full domain
        response = requests.get(f'https://dns.google/resolve?name={domain}&type=MX', timeout=5)
        mx_data = response.json()
        features['has_mx_record'] = int(bool(mx_data.get('Answer', [])))
        features['num_mx_records'] = len(mx_data.get('Answer', []))
        logger.debug(f"MX records found: {features['num_mx_records']}")
        
        # For NS records, try the parent domain if it's a subdomain
        extracted = tldextract.extract(domain)
        parent_domain = f"{extracted.domain}.{extracted.suffix}"  # Get parent domain (e.g., unpar.ac.id from studentportal.unpar.ac.id)
        response = requests.get(f'https://dns.google/resolve?name={parent_domain}&type=NS', timeout=5)
        ns_data = response.json()
        features['has_ns_record'] = int(bool(ns_data.get('Answer', [])))
        features['num_ns_records'] = len(ns_data.get('Answer', []))
        logger.debug(f"NS records found for {parent_domain}: {features['num_ns_records']}")
    except Exception as e:
        logger.error(f"Failed to get MX/NS records using DNS API: {str(e)}")
        features['has_mx_record'] = 0
        features['num_mx_records'] = 0
        features['has_ns_record'] = 0
        features['num_ns_records'] = 0
    
    return features

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
            
        # Get DNS information using alternative approach
        dns_info = get_dns_info(domain)
        features.update(dns_info)
        
        # WHOIS Information
        try:
            logger.debug(f"Attempting WHOIS lookup for {domain}")
            whois_info = whois.whois(domain)
            
            # Creation date
            creation_date = whois_info.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            
            logger.debug(f"Creation date found: {creation_date}")
                
            if creation_date:
                domain_age = (datetime.now() - creation_date).days
                features['domain_age_days'] = domain_age
                features['is_domain_young'] = int(domain_age < 365)  # Domain less than 1 year old
                logger.debug(f"Domain age calculated: {domain_age} days")
            else:
                logger.warning(f"No creation date found for {domain}")
                features['domain_age_days'] = -1
                features['is_domain_young'] = 1
                
            # Expiration date
            expiration_date = whois_info.expiration_date
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]
            
            logger.debug(f"Expiration date found: {expiration_date}")
                
            if expiration_date:
                days_to_expiration = (expiration_date - datetime.now()).days
                features['days_to_expiration'] = days_to_expiration
                features['is_expiring_soon'] = int(days_to_expiration < 30)
                logger.debug(f"Days to expiration: {days_to_expiration}")
            else:
                logger.warning(f"No expiration date found for {domain}")
                features['days_to_expiration'] = -1
                features['is_expiring_soon'] = 1
                
            # Registration details
            features['has_registrar'] = int(bool(whois_info.registrar))
            features['has_registrant'] = int(bool(getattr(whois_info, 'registrant', None)))
            logger.debug(f"Registrar: {whois_info.registrar}, Registrant: {getattr(whois_info, 'registrant', None)}")
            
        except Exception as e:
            logger.error(f"WHOIS lookup failed for {domain}: {str(e)}")
            features['domain_age_days'] = -1
            features['is_domain_young'] = 1
            features['days_to_expiration'] = -1
            features['is_expiring_soon'] = 1
            features['has_registrar'] = 0
            features['has_registrant'] = 0
            
        # SSL Certificate Check
        if parsed.scheme == 'https':
            try:
                logger.debug(f"Checking SSL certificate for {domain}")
                context = ssl.create_default_context()
                with socket.create_connection((domain, 443)) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        cert = ssock.getpeercert()
                        
                        # Get certificate details
                        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y GMT')
                        not_before = datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y GMT')
                        
                        features['ssl_days_valid'] = (not_after - datetime.now()).days
                        features['ssl_is_valid'] = int(datetime.now() > not_before and datetime.now() < not_after)
                        features['ssl_is_expired'] = int(datetime.now() > not_after)
                        
                        # Check if self-signed by comparing issuer and subject
                        issuer = cert.get('issuer', [])
                        subject = cert.get('subject', [])
                        features['ssl_is_self_signed'] = int(issuer == subject)
                        
                        logger.debug(f"SSL certificate valid from {not_before} to {not_after}")
                        logger.debug(f"SSL self-signed: {features['ssl_is_self_signed']}")
                
            except ssl.SSLError as e:
                logger.error(f"SSL Error for {domain}: {str(e)}")
                features['ssl_days_valid'] = -1
                features['ssl_is_valid'] = 0
                features['ssl_is_expired'] = 1
                features['ssl_is_self_signed'] = 1
            except Exception as e:
                logger.error(f"Error checking SSL for {domain}: {str(e)}")
                features['ssl_days_valid'] = -1
                features['ssl_is_valid'] = 0
                features['ssl_is_expired'] = 1
                features['ssl_is_self_signed'] = 1
        else:
            logger.warning(f"No HTTPS for {domain}")
            features['ssl_days_valid'] = -1
            features['ssl_is_valid'] = 0
            features['ssl_is_expired'] = 1
            features['ssl_is_self_signed'] = 1
            
        # Convert to DataFrame
        return pd.DataFrame([features])
        
    except Exception as e:
        print(f"Error extracting DNS features for URL {url}: {str(e)}")
        return None

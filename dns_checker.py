import dns.resolver
import socket
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any
import whois
import json

class DNSChecker:
    def __init__(self, domain: str):
        self.domain = domain
        self.resolver = dns.resolver.Resolver()
        # Use Google's DNS servers for reliability
        self.resolver.nameservers = ['8.8.8.8', '8.8.4.4']
        
    def get_record(self, record_type: str) -> List[str]:
        """Get specific DNS record type."""
        try:
            answers = self.resolver.resolve(self.domain, record_type)
            return [str(answer) for answer in answers]
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.NoNameservers):
            return []
        except Exception as e:
            print(f"Error getting {record_type} record: {str(e)}")
            return []
    
    def get_ip_info(self, ip: str) -> Dict[str, Any]:
        """Get additional information about an IP address."""
        try:
            hostname = socket.gethostbyaddr(ip)[0]
            return {
                "hostname": hostname,
                "ip": ip
            }
        except:
            return {
                "hostname": "Not found",
                "ip": ip
            }
    
    def get_whois_info(self) -> Dict[str, Any]:
        """Get WHOIS information for the domain."""
        try:
            w = whois.whois(self.domain)
            return {
                "registrar": w.registrar,
                "creation_date": str(w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date),
                "expiration_date": str(w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date),
                "name_servers": w.name_servers if isinstance(w.name_servers, list) else [w.name_servers] if w.name_servers else []
            }
        except Exception as e:
            print(f"Error getting WHOIS info: {str(e)}")
            return {}
    
    def check_all_records(self) -> Dict[str, Any]:
        """Check all relevant DNS records."""
        record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'SOA', 'CNAME']
        results = {
            "domain": self.domain,
            "check_time": datetime.now().isoformat(),
            "records": {},
            "whois_info": self.get_whois_info()
        }
        
        # Get all record types
        for record_type in record_types:
            records = self.get_record(record_type)
            results["records"][record_type] = records
            
            # Get additional IP info for A records
            if record_type == 'A' and records:
                results["ip_info"] = [self.get_ip_info(ip) for ip in records]
        
        return results

def save_results(results: Dict[str, Any], filename: str):
    """Save results to both JSON and CSV formats."""
    # Save as JSON
    json_file = f"{filename}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create a flattened version for CSV
    flattened_data = []
    for record_type, records in results['records'].items():
        for record in records:
            flattened_data.append({
                'domain': results['domain'],
                'check_time': results['check_time'],
                'record_type': record_type,
                'record_value': record
            })
    
    # Save as CSV
    df = pd.DataFrame(flattened_data)
    csv_file = f"{filename}.csv"
    df.to_csv(csv_file, index=False)
    
    return json_file, csv_file

def main():
    domain = "studentportal.unpar.ac.id"
    checker = DNSChecker(domain)
    
    print(f"\nChecking DNS records for {domain}...")
    results = checker.check_all_records()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"dns_check_{domain.replace('.', '_')}_{timestamp}"
    json_file, csv_file = save_results(results, filename)
    
    # Print summary
    print("\nDNS Check Results:")
    print("-" * 50)
    print(f"Domain: {domain}")
    print(f"Check Time: {results['check_time']}")
    print("\nRecords found:")
    for record_type, records in results['records'].items():
        if records:
            print(f"\n{record_type} Records:")
            for record in records:
                print(f"  - {record}")
    
    if 'ip_info' in results:
        print("\nIP Information:")
        for ip_info in results['ip_info']:
            print(f"  - IP: {ip_info['ip']}")
            print(f"    Hostname: {ip_info['hostname']}")
    
    if results['whois_info']:
        print("\nWHOIS Information:")
        for key, value in results['whois_info'].items():
            print(f"  - {key}: {value}")
    
    print(f"\nResults saved to:")
    print(f"- JSON: {json_file}")
    print(f"- CSV: {csv_file}")

if __name__ == "__main__":
    main()

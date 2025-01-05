import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_screenshot_dir():
    """Setup screenshots directory if it doesn't exist"""
    screenshot_dir = Path("static/screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    return screenshot_dir

def get_safe_filename(url):
    """Convert URL to safe filename"""
    # Remove scheme and special characters
    filename = url.replace('http://', '').replace('https://', '').replace('/', '_')
    filename = ''.join(c if c.isalnum() or c in '._- ' else '_' for c in filename)
    return filename[:100] + '.png'  # Limit filename length

def capture_website_screenshot(url, timeout=20):
    """
    Capture a screenshot of the website using Selenium
    
    Args:
        url (str): URL to capture
        timeout (int): Maximum time to wait for page load in seconds
        
    Returns:
        tuple: (success (bool), filepath or error message (str))
    """
    try:
        screenshot_dir = setup_screenshot_dir()
        
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in headless mode
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        # Initialize webdriver
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        
        try:
            # Set page load timeout
            driver.set_page_load_timeout(timeout)
            
            # Navigate to URL
            logger.info(f"Navigating to {url}")
            driver.get(url)
            
            # Wait for page to load
            time.sleep(3)  # Give extra time for dynamic content
            
            # Generate filename and path
            filename = get_safe_filename(url)
            filepath = screenshot_dir / filename
            
            # Capture screenshot
            driver.save_screenshot(str(filepath))
            
            # Return relative path for web serving
            relative_path = f"screenshots/{filename}"
            return True, relative_path
            
        finally:
            driver.quit()
            
    except Exception as e:
        logger.error(f"Screenshot capture failed for {url}: {str(e)}")
        return False, str(e)

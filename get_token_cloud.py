"""
Automated Kite Connect token generation using Selenium.
Designed for cloud environments (Google Colab, etc.) where manual token generation is challenging.
"""
import urllib.parse
import time
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from kiteconnect import KiteConnect

# Import centralized config
from config import (
    API_KEY, API_SECRET, USER_ID, PASSWORD, LOGIN_URL,
    save_tokens, validate_credentials
)


def install_colab_dependencies():
    """Install necessary packages for Colab (run once)"""
    print("Installing dependencies for Colab...")
    os.system("apt-get update")
    os.system("apt install -y chromium-chromedriver")
    os.system("pip install selenium kiteconnect")
    print("Dependencies installed!")


def setup_chrome_driver():
    """Setup Chrome driver for Colab environment"""
    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument('--headless')  # Disabled - Run with visible browser for debugging
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    print("Initializing Chrome driver...")
    driver = webdriver.Chrome(options=chrome_options)
    print("✓ Chrome driver initialized")
    return driver


def automated_login():
    """Automate the Kite login process and capture request_token"""
    print("--- AUTOMATED KITE LOGIN ---")
    
    driver = None
    try:
        driver = setup_chrome_driver()
        print(f"Opening login page: {LOGIN_URL}")
        driver.get(LOGIN_URL)
        
        # Wait for and fill user ID
        print("Entering credentials...")
        user_id_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "userid"))
        )
        user_id_field.send_keys(USER_ID)
        
        # Fill password
        password_field = driver.find_element(By.ID, "password")
        password_field.send_keys(PASSWORD)
        
        # Click login button
        login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
        login_button.click()
        
        # Wait for either PIN page OR redirect with request_token
        print("Waiting for PIN page or redirect...")
        
        redirect_url = None
        
        # Wait and check for redirect or PIN page
        try:
            # Wait up to 30 seconds for either redirect URL with request_token OR PIN page
            WebDriverWait(driver, 30).until(
                lambda d: 'request_token' in d.current_url or len(d.find_elements(By.ID, "pin")) > 0
            )
            
            # Check if we got redirected with request_token (no PIN required)
            if 'request_token' in driver.current_url:
                print("✓ Login successful! No PIN required - directly redirected.")
                redirect_url = driver.current_url
                print(f"Captured URL: {redirect_url}")
            else:
                # PIN page appeared
                pin_field = driver.find_element(By.ID, "pin")
                print("✓ PIN page loaded successfully!")
                
                # Prompt user for PIN
                print("\n⚠️  A PIN has been sent to your registered mobile/email.")
                pin = input("Enter your Kite PIN: ").strip()
                
                # Enter PIN
                print("Entering PIN...")
                pin_field.send_keys(pin)
                
                # Click continue button
                continue_button = driver.find_element(By.XPATH, "//button[@type='submit']")
                continue_button.click()
                
                # Wait for redirect and capture URL
                print("Waiting for redirect...")
                try:
                    WebDriverWait(driver, 30).until(
                        lambda d: 'request_token' in d.current_url
                    )
                    redirect_url = driver.current_url
                    print(f"✓ Redirect successful!")
                    print(f"Captured URL: {redirect_url}")
                except TimeoutException:
                    print("✗ Timeout waiting for redirect URL with request_token")
                    print(f"Current URL: {driver.current_url}")
                    driver.save_screenshot("redirect_error.png")
                    return None
                    
        except TimeoutException:
            print("✗ Timeout waiting for PIN page or redirect")
            print(f"Current URL: {driver.current_url}")
            driver.save_screenshot("timeout_error.png")
            return None
        
        # If redirect_url is still None, something went wrong
        if not redirect_url:
            print("✗ Failed to capture redirect URL")
            return None
        
        # Parse the URL to extract request_token
        parsed = urllib.parse.urlparse(redirect_url)
        params = urllib.parse.parse_qs(parsed.query)
        
        if 'request_token' in params:
            request_token = params['request_token'][0]
            print(f"\n✓ Request Token CAPTURED: {request_token}")
            return request_token
        else:
            print("✗ Could not find request_token in redirect URL")
            return None
            
    except Exception as e:
        print(f"✗ Error during automated login: {str(e)}")
        return None
    finally:
        if driver:
            driver.quit()


def generate_access_token(request_token):
    """Generate access token from request_token"""
    try:
        print("\Saving access token...")
        kite = KiteConnect(api_key=API_KEY)
        
        # Generate session
        client = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = client["access_token"]
        
        print(f"✓ Access Token: {access_token}")
        
        # Save tokens using centralized config
        save_tokens(request_token, access_token)
        
        return access_token
        
    except Exception as e:
        print(f"✗ Error generating access token: {str(e)}")
        return None


def get_token_automated():
    """Main function to automate entire flow"""
    print("=== AUTOMATED KITE TOKEN GENERATION ===\n")
    
    # Validate credentials
    if not validate_credentials():
        return None
    
    # Step 1: Automated login to get request_token
    request_token = automated_login()
    
    if not request_token:
        print("\n✗ Failed to get request token")
        return None
    
    # Step 2: Generate access token
    access_token = generate_access_token(request_token)
    
    if access_token:
        print("\n✓✓✓ SUCCESS! You can now use the access token for trading.")
        return access_token
    else:
        print("\n✗ Failed to generate access token")
        return None


if __name__ == "__main__":
    access_token = get_token_automated()

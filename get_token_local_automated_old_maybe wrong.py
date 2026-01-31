import urllib.parse
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from kiteconnect import KiteConnect

# --- CONFIGURATION ---
API_KEY = os.getenv("KITE_API_KEY", "YOUR_API_KEY_HERE")
API_SECRET = os.getenv("KITE_API_SECRET", "YOUR_API_SECRET_HERE")
USER_ID = os.getenv("KITE_USER_ID", "YOUR_USER_ID")
PASSWORD = os.getenv("KITE_PASSWORD", "YOUR_PASSWORD")
PIN = os.getenv("KITE_PIN", "YOUR_PIN")

LOGIN_URL = f"https://kite.zerodha.com/connect/login?v=3&api_key={API_KEY}"
ENV_FILE_PATH = ".env_access"  # Save access token here

def setup_chrome_driver(headless=True):
    """Setup Chrome driver for local Windows environment"""
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    
    chrome_options = webdriver.ChromeOptions()
    if headless:
        chrome_options.add_argument('--headless')  # Run in background
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    
    # Use webdriver-manager to automatically download and manage chromedriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def automated_login(headless=True):
    """Automate the Kite login process and capture request_token"""
    print("--- AUTOMATED KITE LOGIN ---")
    
    driver = None
    try:
        driver = setup_chrome_driver(headless=headless)
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
        
        # Wait for PIN page and enter PIN
        print("Entering PIN...")
        pin_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "pin"))
        )
        pin_field.send_keys(PIN)
        
        # Click continue button
        continue_button = driver.find_element(By.XPATH, "//button[@type='submit']")
        continue_button.click()
        
        # Wait for redirect and capture URL
        print("Waiting for redirect...")
        time.sleep(3)  # Wait for redirect
        
        redirect_url = driver.current_url
        print(f"Captured URL: {redirect_url}")
        
        # Parse the URL to extract request_token
        parsed = urllib.parse.urlparse(redirect_url)
        params = urllib.parse.parse_qs(parsed.query)
        
        if 'request_token' in params:
            request_token = params['request_token'][0]
            print(f"✓ Request Token: {request_token}")
            return request_token
        else:
            print("✗ Could not find request_token in redirect URL")
            return None
            
    except TimeoutException:
        print("✗ Timeout waiting for page elements. Check credentials.")
        return None
    except Exception as e:
        print(f"✗ Error during automated login: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if driver:
            driver.quit()

def generate_access_token(request_token):
    """Generate access token from request_token"""
    try:
        print("\nGenerating access token...")
        kite = KiteConnect(api_key=API_KEY)
        
        # Generate session
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        
        print(f"✓ Access Token: {access_token}")
        
        # Save to .env_access file
        with open(ENV_FILE_PATH, "w") as f:
            f.write(f"KITE_ACCESS_TOKEN={access_token}\n")
            f.write(f"KITE_API_KEY={API_KEY}\n")
        
        print(f"✓ Access token saved to {ENV_FILE_PATH}")
        return access_token
        
    except Exception as e:
        print(f"✗ Error generating access token: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_token_automated(headless=True):
    """Main function to automate entire flow"""
    print("=== AUTOMATED KITE TOKEN GENERATION (LOCAL) ===\n")
    
    # Check if credentials are set
    if API_KEY == "YOUR_API_KEY_HERE" or USER_ID == "YOUR_USER_ID":
        print("⚠ Please set your credentials in environment variables:")
        print("  - KITE_API_KEY")
        print("  - KITE_API_SECRET")
        print("  - KITE_USER_ID")
        print("  - KITE_PASSWORD")
        print("  - KITE_PIN")
        print("\nOr set them in code before running.")
        return None
    
    # Step 1: Automated login to get request_token
    request_token = automated_login(headless=headless)
    
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
    # Run with headless=False to see the browser (helpful for debugging)
    access_token = get_token_automated(headless=True)

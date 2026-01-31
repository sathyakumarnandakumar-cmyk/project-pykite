import os
import webbrowser
from flask import Flask, request
from kiteconnect import KiteConnect
from dotenv import load_dotenv, set_key
import threading

# --- CONFIGURATION ---
ENV_FILE_PATH = ".env_access"

app = Flask(__name__)
request_token_captured = None

def get_token_web_simple():
    """
    Simple web-based token retrieval using Flask server.
    Opens browser automatically and captures token via callback.
    """
    global request_token_captured
    
    print("--- SIMPLE WEB KITE CONNECT LOGIN ---\n")
    
    # Load environment variables
    load_dotenv(ENV_FILE_PATH)
    
    API_KEY = os.environ.get('KITE_API_KEY')
    API_SECRET = os.environ.get('KITE_API_SECRET')
    REDIRECT_URI = os.environ.get('KITE_REDIRECT_URI', 'http://localhost:5000/auth/callback')
    
    if not API_KEY:
        raise ValueError("KITE_API_KEY environment variable must be set")
    if not API_SECRET:
        raise ValueError("KITE_API_SECRET environment variable must be set")
    
    print(f"API_KEY: {API_KEY}")
    print(f"Redirect URI: {REDIRECT_URI}\n")
    
    # Initialize Kite Connect
    kite = KiteConnect(api_key=API_KEY)
    login_url = kite.login_url()
    
    # Define callback route
    @app.route('/auth/callback')
    def callback():
        global request_token_captured
        request_token_captured = request.args.get('request_token')
        
        if request_token_captured:
            return """
            <html>
                <head><title>Kite Login Success</title></head>
                <body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1 style="color: green;">✓ Authentication Successful!</h1>
                    <p>Request token captured. Processing...</p>
                    <p>You can close this window.</p>
                </body>
            </html>
            """
        else:
            return """
            <html>
                <head><title>Kite Login Failed</title></head>
                <body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1 style="color: red;">✗ Authentication Failed</h1>
                    <p>No request token found in callback.</p>
                </body>
            </html>
            """
    
    # Start Flask server in background thread
    def run_server():
        app.run(port=5000, debug=False, use_reloader=False)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    print("Flask server started on http://localhost:5000")
    print("Opening login page in browser...\n")
    
    # Open browser automatically
    webbrowser.open(login_url)
    
    print("Waiting for you to log in through the browser...")
    print("(The browser will redirect back to this application)\n")
    
    # Wait for request token (timeout after 2 minutes)
    import time
    timeout = 120
    elapsed = 0
    while request_token_captured is None and elapsed < timeout:
        time.sleep(1)
        elapsed += 1
    
    if request_token_captured is None:
        print("\n[ERROR] Timeout waiting for login. Please try again.")
        return None
    
    print(f"[SUCCESS] Request Token captured: {request_token_captured}")
    
    # Save request token
    set_key(ENV_FILE_PATH, "KITE_REQUEST_TOKEN", request_token_captured)
    print(f"[SAVED] Request Token saved to {ENV_FILE_PATH}")
    
    # Generate access token
    try:
        session_data = kite.generate_session(request_token_captured, api_secret=API_SECRET)
        access_token = session_data["access_token"]
        
        print(f"\n[SUCCESS] Access Token generated: {access_token}")
        
        # Save access token
        set_key(ENV_FILE_PATH, "KITE_ACCESS_TOKEN", access_token)
        print(f"[SAVED] Access Token saved to {ENV_FILE_PATH}")
        
        print("\n✓ Authentication complete! Access token is ready for trading.")
        return access_token
        
    except Exception as e:
        print(f"\n[ERROR] Failed to generate session: {str(e)}")
        return None

if __name__ == "__main__":
    access_token = get_token_web_simple()
    
    if access_token:
        print("\n" + "="*60)
        print("SUCCESS! Your access token is ready.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("FAILED! Please check the errors above and try again.")
        print("="*60)

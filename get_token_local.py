"""
Local Kite Connect token generation using HTTP server.
Opens browser for login and captures the redirect callback.
"""
import http.server
import socketserver
import urllib.parse
import logging
import webbrowser

from kiteconnect import KiteConnect

# Import centralized config
from config import (
    API_KEY, API_SECRET, PORT, LOGIN_URL,
    save_tokens, validate_credentials
)

logging.basicConfig(level=logging.DEBUG)

# Initialize Kite
kite = KiteConnect(api_key=API_KEY)

if API_KEY:
    print(f"API_KEY: {API_KEY[:5]}...")
else:
    print("API_KEY: None")
print(f"Login URL: {LOGIN_URL}")


class TokenHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler to capture the redirect callback with request_token"""
    
    def do_GET(self):
        # Parse the URL to get the query parameters
        parsed_path = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_path.query)

        # Check if request_token exists in the parameters
        if 'request_token' in query_params:
            request_token = query_params['request_token'][0]
            
            print(f"\n[SUCCESS] Request Token captured: {request_token}")
            
            try:
                # Generate session and get access token
                client = kite.generate_session(request_token, api_secret=API_SECRET)
                access_token = client["access_token"]
                
                # Print access token preview (first 5 characters for security)
                print(f"\n[INFO] Access Token received: {access_token[:5]}...")
                
                # Save tokens using centralized config (it knows where to save)
                save_tokens(request_token, access_token)
                print(f"[INFO] Tokens saved via config.save_tokens()")
                
                # Send success message to browser
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<h1>Success! Token captured and saved. You can close this window.</h1>")
                
            except Exception as e:
                print(f"[ERROR] Failed to generate access token: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f"<h1>Error: {e}</h1>".encode())
            
            # Stop the server
            global server_keep_running
            server_keep_running = False
        else:
            # Handle cases where the URL is wrong
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"<h1>Error: No request_token found in URL.</h1>")



def login():
    """
    Triggers the local login flow.
    """
    global server_keep_running
    
    # Validate credentials
    if not validate_credentials():
        return
    
    # Open the login page automatically in default browser
    print(f"Opening login page: {LOGIN_URL}")
    webbrowser.open(LOGIN_URL)

    # Start the local server to listen for the redirect
    print(f"Listening for redirect on http://127.0.0.1:{PORT}...")
    
    server_keep_running = True
    with socketserver.TCPServer(("", PORT), TokenHandler) as httpd:
        # Set a timeout so the loop yields control periodically, allowing interrupts
        httpd.timeout = 1.0
        while server_keep_running:
            httpd.handle_request()

    print("\n✓ Server stopped. You can now run your main trading bot.")

if __name__ == "__main__":
    login()

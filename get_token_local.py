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

from config import get_app_config, save_tokens, validate_credentials

logging.basicConfig(level=logging.DEBUG)

# Global variables for the local server state
server_keep_running = False
current_app_name = None
current_kite = None

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
                global current_app_name
                global current_kite
                
                # Fetch app config to get API_SECRET
                app_conf = get_app_config(current_app_name)
                api_secret = app_conf['KITE_API_SECRET']
                
                # Generate session and get access token
                client = current_kite.generate_session(request_token, api_secret=api_secret)
                access_token = client["access_token"]
                
                # Print access token preview
                print(f"\n[INFO] Access Token received: {access_token[:5]}...")
                
                # Save tokens specifically to the correct section
                save_tokens(current_app_name, request_token, access_token)
                
                # Send success message to browser
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f"<h1>Success! Token captured and saved for {current_app_name}. You can close this window.</h1>".encode())
                
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
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"<h1>Error: No request_token found in URL.</h1>")


def login(app_name="Full_App"):
    """
    Triggers the local login flow for a specific app.
    """
    global server_keep_running
    global current_app_name
    global current_kite
    
    # Set the global context for the TokenHandler
    current_app_name = app_name
    
    # 1. Validate credentials
    if not validate_credentials(app_name):
        return
    
    # 2. Fetch the correct configuration
    app_conf = get_app_config(app_name)
    api_key = app_conf["KITE_API_KEY"]
    redirect_uri = app_conf.get("KITE_REDIRECT_URI", "http://127.0.0.1:5010")
    
    # Extract port dynamically from the app's redirect URI
    parsed_uri = urllib.parse.urlparse(redirect_uri)
    port = parsed_uri.port if parsed_uri.port else 80
    
    # 3. Initialize Kite temporarily to generate the session later
    current_kite = KiteConnect(api_key=api_key)
    
    print(f"\n--- Initiating Login for App: {app_name} ---")
    print(f"API_KEY: {api_key[:5]}...")
    
    # Generate login URL dynamically
    login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
    print(f"Opening login page: {login_url}")
    webbrowser.open(login_url)

    # Start the local server
    print(f"Listening for redirect on http://127.0.0.1:{port}...")
    
    server_keep_running = True
    with socketserver.TCPServer(("", port), TokenHandler) as httpd:
        httpd.timeout = 1.0
        while server_keep_running:
            httpd.handle_request()

    print(f"\n✓ Server stopped. Authentication complete for {app_name}.")

if __name__ == "__main__":
    # Add parent dir to path so 'config' always evaluates correctly
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    app_to_test = sys.argv[1] if len(sys.argv) > 1 else "Full_App"
    print(f"ℹ️ Testing '{app_to_test}'. You can pass an app name like: `python pykite/deploy/get_token_local.py algotestlive`")
    login(app_to_test)

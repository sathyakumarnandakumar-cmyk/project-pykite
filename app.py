from flask import Flask, redirect, request, url_for, session
from kiteconnect import KiteConnect
import os
import secrets # For generating a secure secret key

app = Flask(__name__)

# IMPORTANT: Use a strong, random secret key for Flask sessions.
# In production, get this from environment variables.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(16))

# --- Retrieve your API Key and API Secret from environment variables ---
# NEVER hardcode these in your script, especially for production!
API_KEY = os.environ.get('KITE_API_KEY')
API_SECRET = os.environ.get('KITE_API_SECRET')
REDIRECT_URI = os.environ.get('KITE_REDIRECT_URI', 'http://localhost:5000/auth/callback') # Must match what you set in Kite Dev Console

if not API_KEY or not API_SECRET:
    raise ValueError("KITE_API_KEY and KITE_API_SECRET environment variables must be set.")

kite = KiteConnect(api_key=API_KEY)

# Store kite instance and other temporary auth data in session
# This is a simple way for demo; for production, consider a more robust state management
# You might also want to store API_KEY, API_SECRET, REDIRECT_URI somewhere safe.

@app.route('/')
def home():
    if 'access_token' in session:
        return f"<h1>Kite Connected!</h1><p>Access Token: {session['access_token']}</p><p>You can now run your main script.</p><p><a href='/logout'>Logout</a></p>"
    else:
        # Generate the login URL
        login_url = kite.login_url()
        return f"<h1>Welcome to Kite Connect Demo</h1><p><a href='{login_url}'>Login via Kite</a></p>"

@app.route('/auth/callback')
def auth_callback():
    request_token = request.args.get('request_token')
    if not request_token:
        return "Error: request_token not found in callback.", 400

    try:
        # Generate access token
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        public_token = data["public_token"]
        user_id = data["user_id"]

        session['access_token'] = access_token
        session['public_token'] = public_token
        session['user_id'] = user_id

        print(f"Successfully obtained access_token: {access_token}")
        print(f"User ID: {user_id}")

        return redirect(url_for('home'))

    except Exception as e:
        return f"Error during token exchange: {e}", 500

@app.route('/logout')
def logout():
    session.pop('access_token', None)
    session.pop('public_token', None)
    session.pop('user_id', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    # Set environment variables before running:
    # export KITE_API_KEY="YOUR_API_KEY"
    # export KITE_API_SECRET="YOUR_API_SECRET"
    # export KITE_REDIRECT_URI="http://localhost:5000/auth/callback" # Or whatever you set
    # export FLASK_SECRET_KEY="a_very_secret_random_string" # Generate with secrets.token_hex(16)

    print(f"Kite API Key: {API_KEY}")
    print(f"Kite Redirect URI: {REDIRECT_URI}")
    print(f"Flask Secret Key: {app.secret_key[:5]}...")

    # Set the redirect URL for the KiteConnect instance
    kite.set_access_token(None) # Clear any previous access token if set
    #kite.set_redirect_url(REDIRECT_URI)

    # Run the Flask app
    app.run(debug=True) # debug=True is for development only!
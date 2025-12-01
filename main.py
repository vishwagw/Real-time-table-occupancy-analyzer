import webview
import threading
import time
import sys
import os

# Add the backend to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app import app as flask_app

class Api:
    def __init__(self):
        self.window = None

    def set_window(self, window):
        self.window = window

    def exit_app(self):
        if self.window:
            self.window.destroy()

def run_flask():
    """Run Flask server in a separate thread"""
    flask_app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    # Start Flask server in a background thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    # Create API instance
    api = Api()
    
    # Create webview window
    window = webview.create_window(
        'Table Occupancy Detection System',
        'http://127.0.0.1:5000',
        width=1200,
        height=800,
        min_size=(800, 600),
        js_api=api
    )
    
    api.set_window(window)
    
    # Start the application
    webview.start(debug=True)
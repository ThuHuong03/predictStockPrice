"""
Script để chạy web server trong chế độ test
"""

import os
import sys
from pathlib import Path

# Change to correct directory
CURRENT_DIR = Path(__file__).parent
os.chdir(CURRENT_DIR)

# Import and start web app
from web_app import app, initialize_model

print("🚀 Starting web server for testing...")

# Initialize model
success = initialize_model()

if success:
    print("✅ Model loaded successfully!")
    print("🌐 Starting Flask server on http://127.0.0.1:5000")
    print("📊 Dashboard: http://127.0.0.1:5000/dashboard")
    print("🔗 API: http://127.0.0.1:5000/api/stocks")

    # Run Flask app with no debug for testing
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)
else:
    print("❌ Failed to initialize model!")
    sys.exit(1)
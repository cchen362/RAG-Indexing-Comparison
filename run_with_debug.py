"""
Script to run the Streamlit app with debug logging
Usage: python run_with_debug.py
"""

import subprocess
import sys
import os
from logger_config import setup_logging
import logging

def main():
    print("🐛 Starting RAG App with DEBUG logging...")
    
    # Setup debug logging
    setup_logging(log_level=logging.DEBUG, log_to_file=True)
    
    # Set environment variable for debug mode
    os.environ['RAG_DEBUG'] = 'true'
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--logger.level=debug"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Stopping app...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()
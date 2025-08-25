#!/usr/bin/env python3
"""
Google Sheets Credentials Encoder
Encodes the Google Sheets service account JSON file to base64 for environment variable injection.
"""
import base64
import json
import os

def encode_credentials():
    """Encode Google Sheets credentials to base64"""
    creds_file = "credentials/rag-comparison-app-2ebc99e885e4.json"
    
    if not os.path.exists(creds_file):
        print(f"FAIL: Credentials file not found: {creds_file}")
        print("Please ensure your Google Sheets service account JSON is in the credentials/ folder")
        return None
    
    try:
        # Read the JSON file
        with open(creds_file, 'r') as f:
            creds_data = json.load(f)
        
        # Convert back to JSON string (ensures proper formatting)
        creds_json = json.dumps(creds_data)
        
        # Encode to base64
        creds_b64 = base64.b64encode(creds_json.encode('utf-8')).decode('utf-8')
        
        print("SUCCESS: Google Sheets credentials encoded successfully!")
        print("\nAdd this to your .env file:")
        print(f"GOOGLE_CREDENTIALS_BASE64={creds_b64}")
        
        # Also write to a separate env file for easy copying
        with open('.env.credentials', 'w') as f:
            f.write(f"GOOGLE_CREDENTIALS_BASE64={creds_b64}\n")
        
        print(f"\nCredentials also saved to: .env.credentials")
        print("You can copy this value to your production .env file")
        
        return creds_b64
        
    except json.JSONDecodeError as e:
        print(f"FAIL: Invalid JSON in credentials file: {e}")
        return None
    except Exception as e:
        print(f"FAIL: Failed to encode credentials: {e}")
        return None

if __name__ == "__main__":
    encode_credentials()
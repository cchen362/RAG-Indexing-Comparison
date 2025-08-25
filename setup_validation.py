#!/usr/bin/env python3
"""
Setup validation script for RAG Indexing Comparison App
Tests API keys and Google Sheets connection before main development
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import openai
import cohere
import gspread
from google.oauth2.service_account import Credentials

def load_environment():
    """Load environment variables from .env file"""
    env_path = Path('.env')
    if not env_path.exists():
        print("[FAIL] .env file not found. Please create one with your API keys.")
        return False
    
    load_dotenv()
    return True

def test_openai_api():
    """Test OpenAI API connection"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("[FAIL] OPENAI_API_KEY not found in .env file")
        return False
    
    try:
        client = openai.OpenAI(api_key=api_key)
        # Test with a simple embedding call
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="test"
        )
        print("[OK] OpenAI API connection successful")
        print(f"   Embedding dimension: {len(response.data[0].embedding)}")
        return True
    except Exception as e:
        print(f"[FAIL] OpenAI API test failed: {str(e)}")
        return False

def test_cohere_api():
    """Test Cohere API connection"""
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        print("[FAIL] COHERE_API_KEY not found in .env file")
        return False
    
    try:
        co = cohere.ClientV2(api_key=api_key)
        # Test with a simple embedding call  
        response = co.embed(
            texts=["test"],
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"]
        )
        print("[OK] Cohere API connection successful")
        print(f"   Embedding dimension: {len(response.embeddings.float_[0])}")
        return True
    except Exception as e:
        print(f"[FAIL] Cohere API test failed: {str(e)}")
        return False

def test_google_sheets():
    """Test Google Sheets API connection"""
    creds_path = Path('credentials/rag-comparison-app-2ebc99e885e4.json')
    if not creds_path.exists():
        print(f"[FAIL] Google credentials file not found at {creds_path}")
        return False
    
    sheet_id = os.getenv('GOOGLE_SHEET_ID', '1U34uloZe1S0E-T83LDOtKfgYuBipBrejGdEW8QVSguI')
    
    try:
        # Load credentials
        scope = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        credentials = Credentials.from_service_account_file(str(creds_path), scopes=scope)
        gc = gspread.authorize(credentials)
        
        # Test sheet access
        sheet = gc.open_by_key(sheet_id)
        worksheet = sheet.sheet1
        
        # Test write access with a simple test
        test_data = [["Test", "Connection", "Success"]]
        worksheet.append_rows(test_data)
        
        print("[OK] Google Sheets API connection successful")
        print(f"   Sheet title: {sheet.title}")
        print(f"   Worksheet: {worksheet.title}")
        return True
    except Exception as e:
        print(f"[FAIL] Google Sheets API test failed: {str(e)}")
        return False

def main():
    """Run all validation tests"""
    print("Starting RAG Indexing Comparison App setup validation...\n")
    
    # Load environment
    if not load_environment():
        sys.exit(1)
    
    # Run tests
    tests = [
        ("OpenAI API", test_openai_api),
        ("Cohere API", test_cohere_api),
        ("Google Sheets API", test_google_sheets)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        result = test_func()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"[OK] All {total} tests passed! Setup is ready for development.")
    else:
        print(f"!  {passed}/{total} tests passed. Please fix the failing tests before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
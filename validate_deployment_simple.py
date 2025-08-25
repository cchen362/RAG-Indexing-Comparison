#!/usr/bin/env python3
"""
Simple Deployment Validation Script (Windows Compatible)
Validates that all prerequisites are in place for successful deployment.
"""
import os
import json
import base64
from pathlib import Path

def validate_all():
    """Run all validation checks"""
    print("RAG Comparison App - Deployment Validation")
    print("=" * 50)
    
    all_good = True
    
    # Check .env file
    if not Path(".env").exists():
        print("[FAIL] .env file not found")
        all_good = False
    else:
        with open(".env", 'r') as f:
            env_content = f.read()
        if "OPENAI_API_KEY" not in env_content or "COHERE_API_KEY" not in env_content:
            print("[FAIL] Missing API keys in .env file")
            all_good = False
        else:
            print("[PASS] .env file is valid")
    
    # Check credentials
    creds_file = Path("credentials/rag-comparison-app-2ebc99e885e4.json")
    if not creds_file.exists():
        print("[FAIL] Google Sheets credentials not found")
        all_good = False
    else:
        try:
            with open(creds_file, 'r') as f:
                creds = json.load(f)
            if creds.get('type') == 'service_account':
                print("[PASS] Google Sheets credentials are valid")
            else:
                print("[FAIL] Invalid credentials format")
                all_good = False
        except:
            print("[FAIL] Cannot read credentials file")
            all_good = False
    
    # Check Docker files
    docker_files = ['Dockerfile', 'docker-compose.yml']
    for file in docker_files:
        if Path(file).exists():
            print(f"[PASS] {file} exists")
        else:
            print(f"[FAIL] {file} missing")
            all_good = False
    
    # Check app files
    app_files = ['app.py', 'requirements.txt']
    for file in app_files:
        if Path(file).exists():
            print(f"[PASS] {file} exists")
        else:
            print(f"[FAIL] {file} missing")
            all_good = False
    
    # Check scripts
    scripts = ['deploy.sh', 'quick_deploy.sh', 'encode_credentials.py']
    script_count = 0
    for script in scripts:
        if Path(script).exists():
            script_count += 1
    
    if script_count > 0:
        print(f"[PASS] Deployment scripts available ({script_count} found)")
    else:
        print("[FAIL] No deployment scripts found")
        all_good = False
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("SUCCESS: All checks passed!")
        print("\nYou can now deploy with:")
        print("1. python encode_credentials.py  # Generate base64 credentials")
        print("2. ./quick_deploy.sh             # Deploy to POP server")
        print("   or")
        print("2. ./deploy.sh                   # Test locally first")
        return True
    else:
        print("FAILED: Fix the issues above before deploying")
        return False

if __name__ == "__main__":
    success = validate_all()
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
Deployment Validation Script
Validates that all prerequisites are in place for successful deployment.
"""
import os
import json
import base64
from pathlib import Path

def check_env_file():
    """Check if .env file exists and has required keys"""
    env_file = Path(".env")
    if not env_file.exists():
        return False, "❌ .env file not found"
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        required_keys = ['OPENAI_API_KEY', 'COHERE_API_KEY']
        missing_keys = []
        
        for key in required_keys:
            if key not in content or f"{key}=" not in content:
                missing_keys.append(key)
        
        if missing_keys:
            return False, f"[FAIL] Missing API keys: {', '.join(missing_keys)}"
        
        return True, "[PASS] .env file is valid"
        
    except Exception as e:
        return False, f"[FAIL] Error reading .env file: {e}"

def check_credentials():
    """Check if Google Sheets credentials exist and are valid"""
    creds_file = Path("credentials/rag-comparison-app-2ebc99e885e4.json")
    
    if not creds_file.exists():
        return False, "❌ Google Sheets credentials not found"
    
    try:
        with open(creds_file, 'r') as f:
            creds_data = json.load(f)
        
        required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
        missing_fields = []
        
        for field in required_fields:
            if field not in creds_data:
                missing_fields.append(field)
        
        if missing_fields:
            return False, f"❌ Invalid credentials JSON, missing fields: {', '.join(missing_fields)}"
        
        if creds_data.get('type') != 'service_account':
            return False, "❌ Credentials must be for a service account"
        
        return True, "✅ Google Sheets credentials are valid"
        
    except json.JSONDecodeError:
        return False, "❌ Invalid JSON in credentials file"
    except Exception as e:
        return False, f"❌ Error reading credentials: {e}"

def check_docker_files():
    """Check if Docker configuration files exist"""
    required_files = ['Dockerfile', 'docker-compose.yml']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        return False, f"❌ Missing Docker files: {', '.join(missing_files)}"
    
    return True, "✅ Docker configuration files exist"

def check_app_files():
    """Check if main application files exist"""
    required_files = ['app.py', 'requirements.txt']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        return False, f"❌ Missing application files: {', '.join(missing_files)}"
    
    return True, "✅ Application files exist"

def check_deployment_scripts():
    """Check if deployment scripts exist"""
    scripts = ['deploy.sh', 'deploy_to_pop.sh', 'quick_deploy.sh']
    existing_scripts = []
    
    for script in scripts:
        if Path(script).exists():
            existing_scripts.append(script)
    
    if not existing_scripts:
        return False, "❌ No deployment scripts found"
    
    return True, f"✅ Deployment scripts available: {', '.join(existing_scripts)}"

def test_credential_encoding():
    """Test the credential encoding process"""
    try:
        creds_file = Path("credentials/rag-comparison-app-2ebc99e885e4.json")
        if not creds_file.exists():
            return False, "❌ Cannot test encoding - credentials file missing"
        
        with open(creds_file, 'r') as f:
            creds_data = json.load(f)
        
        # Convert to JSON string and encode
        creds_json = json.dumps(creds_data)
        creds_b64 = base64.b64encode(creds_json.encode('utf-8')).decode('utf-8')
        
        # Test decoding
        decoded_json = base64.b64decode(creds_b64).decode('utf-8')
        decoded_data = json.loads(decoded_json)
        
        if decoded_data == creds_data:
            return True, "✅ Credential encoding/decoding test passed"
        else:
            return False, "❌ Credential encoding/decoding test failed"
            
    except Exception as e:
        return False, f"❌ Credential encoding test failed: {e}"

def main():
    """Run all validation checks"""
    print("RAG Comparison App - Deployment Validation")
    print("=" * 50)
    
    checks = [
        ("Environment File", check_env_file),
        ("Google Sheets Credentials", check_credentials),
        ("Docker Configuration", check_docker_files),
        ("Application Files", check_app_files),
        ("Deployment Scripts", check_deployment_scripts),
        ("Credential Encoding", test_credential_encoding),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            passed, message = check_func()
            print(f"{check_name}: {message}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"{check_name}: ❌ Unexpected error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("All validation checks passed!")
        print("Ready for deployment")
        print("\nNext steps:")
        print("1. Make scripts executable (if on Linux/Mac): chmod +x *.sh")
        print("2. Deploy with: ./quick_deploy.sh")
        print("3. Or test locally first: ./deploy.sh")
        return 0
    else:
        print("Some validation checks failed")
        print("Please fix the issues above before deploying")
        return 1

if __name__ == "__main__":
    exit(main())
"""
Quick test to verify Cohere API dimension fix
"""

import os
from dotenv import load_dotenv
import cohere
import numpy as np

load_dotenv()

def test_cohere_dimensions():
    """Test Cohere API with different dimensions"""
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        print("❌ COHERE_API_KEY not found")
        return False
    
    try:
        client = cohere.ClientV2(api_key=api_key)
        
        # Test 512 dimensions
        print("Testing Cohere with 512 dimensions...")
        response_512 = client.embed(
            texts=["This is a test"],
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
            dimensions=512
        )
        
        embedding_512 = response_512.embeddings.float_[0]
        print(f"✅ 512-dim embedding shape: {len(embedding_512)}")
        
        # Test 1024 dimensions
        print("Testing Cohere with 1024 dimensions...")
        response_1024 = client.embed(
            texts=["This is a test"],
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
            dimensions=1024
        )
        
        embedding_1024 = response_1024.embeddings.float_[0]
        print(f"✅ 1024-dim embedding shape: {len(embedding_1024)}")
        
        # Test without dimensions parameter (should get default)
        print("Testing Cohere without dimensions parameter...")
        response_default = client.embed(
            texts=["This is a test"],
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"]
        )
        
        embedding_default = response_default.embeddings.float_[0]
        print(f"✅ Default embedding shape: {len(embedding_default)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_cohere_dimensions()
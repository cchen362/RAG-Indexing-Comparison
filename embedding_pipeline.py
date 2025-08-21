"""
Embedding pipeline module for RAG Indexing Comparison App
Handles different embedding models and vector creation
"""

import os
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import openai
import cohere
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

class EmbeddingProvider:
    """Base class for embedding providers"""
    
    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension
    
    def embed_texts(self, texts: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Embed a list of texts
        Returns: (embeddings_array, metadata)
        """
        raise NotImplementedError
    
    def embed_query(self, query: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Embed a single query
        Returns: (embedding_vector, metadata)
        """
        embeddings, metadata = self.embed_texts([query])
        return embeddings[0], metadata


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, model_name: str, dimension: int):
        super().__init__(model_name, dimension)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = openai.OpenAI(api_key=api_key)
    
    def embed_texts(self, texts: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Embed texts using OpenAI API"""
        start_time = time.time()
        
        try:
            # Process in batches to avoid API limits
            batch_size = 100  # OpenAI allows up to 2048 inputs
            all_embeddings = []
            total_tokens = 0
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                total_tokens += response.usage.total_tokens
            
            embeddings_array = np.array(all_embeddings)
            
            # Apply L2 normalization
            embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            
            end_time = time.time()
            
            metadata = {
                'provider': 'openai',
                'model': self.model_name,
                'dimension': self.dimension,
                'processing_time': end_time - start_time,
                'total_tokens': total_tokens,
                'texts_count': len(texts),
                'batch_count': (len(texts) + batch_size - 1) // batch_size
            }
            
            return embeddings_array, metadata
            
        except Exception as e:
            raise Exception(f"OpenAI embedding failed: {str(e)}")


class CohereEmbedding(EmbeddingProvider):
    """Cohere embedding provider"""
    
    def __init__(self, model_name: str, dimension: int):
        super().__init__(model_name, dimension)
        api_key = os.getenv('COHERE_API_KEY')
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in environment")
        self.client = cohere.ClientV2(api_key=api_key)
    
    def embed_texts(self, texts: List[str], input_type: str = "search_document") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Embed texts using Cohere API"""
        start_time = time.time()
        
        try:
            # Process in batches to avoid API limits
            batch_size = 96  # Cohere allows up to 96 texts per request
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Use basic Cohere v5 API call (dimensions handled by model default)
                response = self.client.embed(
                    texts=batch_texts,
                    model=self.model_name,
                    input_type=input_type,
                    embedding_types=["float"]
                )
                
                batch_embeddings = response.embeddings.float_
                all_embeddings.extend(batch_embeddings)
            
            embeddings_array = np.array(all_embeddings)
            
            # Apply L2 normalization
            embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            
            # Verify dimension matches expected
            actual_dimension = embeddings_array.shape[1]
            if actual_dimension != self.dimension:
                st.warning(f"Expected dimension {self.dimension}, got {actual_dimension}")
                # Update dimension to actual for consistency
                self.dimension = actual_dimension
            
            end_time = time.time()
            
            metadata = {
                'provider': 'cohere',
                'model': self.model_name,
                'dimension': embeddings_array.shape[1],
                'processing_time': end_time - start_time,
                'texts_count': len(texts),
                'batch_count': (len(texts) + batch_size - 1) // batch_size,
                'input_type': input_type
            }
            
            return embeddings_array, metadata
            
        except Exception as e:
            raise Exception(f"Cohere embedding failed: {str(e)}")
    
    def embed_query(self, query: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Embed a single query with search_query input type"""
        embeddings, metadata = self.embed_texts([query], input_type="search_query")
        return embeddings[0], metadata


class EmbeddingPipeline:
    """Main embedding pipeline orchestrator"""
    
    def __init__(self):
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available embedding providers"""
        try:
            # OpenAI providers
            self.providers['openai_small'] = OpenAIEmbedding(
                model_name='text-embedding-3-small',
                dimension=1536
            )
            self.providers['openai_large'] = OpenAIEmbedding(
                model_name='text-embedding-3-large', 
                dimension=3072
            )
        except Exception as e:
            st.warning(f"OpenAI embedding provider not available: {str(e)}")
        
        try:
            # Cohere providers - use default dimension (will be updated after first API call)
            self.providers['cohere_default'] = CohereEmbedding(
                model_name='embed-v4.0',
                dimension=1536  # Default dimension for embed-v4.0
            )
        except Exception as e:
            st.warning(f"Cohere embedding provider not available: {str(e)}")
    
    def get_provider(self, config: Dict[str, Any]) -> EmbeddingProvider:
        """Get embedding provider based on configuration"""
        provider = config['embedding']['provider']
        dimension = config['embedding']['dimension']
        
        if provider == 'openai':
            if dimension == 1536:
                return self.providers.get('openai_small')
            elif dimension == 3072:
                return self.providers.get('openai_large')
        elif provider == 'cohere':
            # Return the default Cohere provider (dimensions will be auto-detected)
            return self.providers.get('cohere_default')
        
        raise ValueError(f"No provider found for {provider} with dimension {dimension}")
    
    def create_embeddings(self, chunks: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create embeddings for document chunks"""
        provider = self.get_provider(config)
        
        if not provider:
            raise ValueError(f"Provider not available for config: {config}")
        
        start_time = time.time()
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings
        embeddings, embedding_metadata = provider.embed_texts(texts)
        
        end_time = time.time()
        
        # Combine chunks with their embeddings
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            enriched_chunk = chunk.copy()
            enriched_chunk['embedding'] = embeddings[i]
            enriched_chunk['embedding_metadata'] = {
                'provider': embedding_metadata['provider'],
                'model': embedding_metadata['model'],
                'dimension': embedding_metadata['dimension']
            }
            enriched_chunks.append(enriched_chunk)
        
        return {
            'chunks': enriched_chunks,
            'embeddings_matrix': embeddings,
            'metadata': {
                'provider': embedding_metadata['provider'],
                'model': embedding_metadata['model'],
                'dimension': embedding_metadata['dimension'],
                'total_processing_time': end_time - start_time,
                'embedding_processing_time': embedding_metadata['processing_time'],
                'total_chunks': len(chunks),
                'total_tokens': embedding_metadata.get('total_tokens', 0)
            }
        }
    
    def embed_query(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create embedding for a query"""
        provider = self.get_provider(config)
        
        if not provider:
            raise ValueError(f"Provider not available for config: {config}")
        
        start_time = time.time()
        
        # Create query embedding
        query_embedding, embedding_metadata = provider.embed_query(query)
        
        end_time = time.time()
        
        return {
            'query': query,
            'embedding': query_embedding,
            'metadata': {
                'provider': embedding_metadata['provider'],
                'model': embedding_metadata['model'],
                'dimension': embedding_metadata['dimension'],
                'processing_time': end_time - start_time,
                'total_tokens': embedding_metadata.get('total_tokens', 0)
            }
        }
    
    def calculate_similarity(self, query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and chunk embeddings"""
        # Both embeddings are already L2 normalized, so dot product = cosine similarity
        similarities = np.dot(chunk_embeddings, query_embedding)
        
        return similarities
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available embedding models"""
        models = {}
        
        for key, provider in self.providers.items():
            if provider:
                models[key] = {
                    'provider': provider.__class__.__name__,
                    'model_name': provider.model_name,
                    'dimension': provider.dimension,
                    'available': True
                }
        
        return models


def test_embedding_pipeline():
    """Test function for embedding pipeline"""
    pipeline = EmbeddingPipeline()
    
    # Test data
    test_chunks = [
        {'id': 0, 'text': 'This is a test document about machine learning.'},
        {'id': 1, 'text': 'Natural language processing is a subset of AI.'},
        {'id': 2, 'text': 'Retrieval augmented generation improves AI responses.'}
    ]
    
    test_config = {
        'embedding': {
            'provider': 'openai',
            'model': 'text-embedding-3-small',
            'dimension': 1536
        }
    }
    
    try:
        # Test document embedding
        result = pipeline.create_embeddings(test_chunks, test_config)
        print(f"Created embeddings for {len(result['chunks'])} chunks")
        print(f"Embedding shape: {result['embeddings_matrix'].shape}")
        
        # Test query embedding
        query_result = pipeline.embed_query("What is machine learning?", test_config)
        print(f"Query embedding shape: {query_result['embedding'].shape}")
        
        # Test similarity calculation
        similarities = pipeline.calculate_similarity(
            query_result['embedding'], 
            result['embeddings_matrix']
        )
        print(f"Similarities: {similarities}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_embedding_pipeline()
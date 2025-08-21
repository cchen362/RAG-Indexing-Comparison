"""
Retrieval system module for RAG Indexing Comparison App
Implements vector, BM25, and hybrid retrieval methods
"""

import time
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict, Counter
import math

class VectorRetriever:
    """Vector-based retrieval using FAISS for efficient similarity search"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.is_trained = False
    
    def build_index(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build FAISS index from embedded chunks"""
        start_time = time.time()
        
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in chunks], dtype=np.float32)
        
        # Update dimension if mismatch (handle dynamic dimensions)
        actual_dimension = embeddings.shape[1]
        if actual_dimension != self.dimension:
            self.dimension = actual_dimension
        
        # Create FAISS index (using flat L2 index for simplicity)
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
        
        # Ensure embeddings are contiguous and float32
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store chunks for later retrieval
        self.chunks = chunks
        self.is_trained = True
        
        end_time = time.time()
        
        return {
            'indexing_time': end_time - start_time,
            'total_vectors': len(chunks),
            'dimension': self.dimension,
            'index_type': 'FAISS_IndexFlatIP'
        }
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Search for most similar chunks"""
        if not self.is_trained:
            raise ValueError("Index not built. Call build_index() first.")
        
        start_time = time.time()
        
        # Normalize query embedding
        query_norm = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(query_norm)
        
        # Search
        scores, indices = self.index.search(query_norm, top_k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                chunk = self.chunks[idx].copy()
                chunk['retrieval_rank'] = i + 1
                chunk['retrieval_score'] = float(score)
                chunk['retrieval_method'] = 'vector'
                results.append(chunk)
        
        end_time = time.time()
        
        metadata = {
            'search_time': end_time - start_time,
            'query_dimension': query_embedding.shape[0],
            'results_count': len(results),
            'method': 'vector'
        }
        
        return results, metadata


class BM25Retriever:
    """BM25 (Best Matching 25) retrieval for keyword-based search"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Field length normalization parameter
        self.chunks = []
        self.vectorizer = None
        self.term_frequencies = None
        self.doc_lengths = None
        self.avg_doc_length = 0
        self.is_trained = False
    
    def _preprocess_text(self, text: str) -> str:
        """Simple text preprocessing for BM25"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return self._preprocess_text(text).split()
    
    def build_index(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build BM25 index from text chunks"""
        start_time = time.time()
        
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        self.chunks = chunks
        
        # Extract and preprocess texts
        texts = [self._preprocess_text(chunk['text']) for chunk in chunks]
        
        # Calculate document lengths
        self.doc_lengths = [len(self._tokenize(text)) for text in texts]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        
        # Build vocabulary and term frequencies
        vocabulary = set()
        doc_term_frequencies = []
        
        for text in texts:
            tokens = self._tokenize(text)
            term_freq = Counter(tokens)
            doc_term_frequencies.append(term_freq)
            vocabulary.update(tokens)
        
        self.vocabulary = sorted(vocabulary)
        self.vocab_size = len(self.vocabulary)
        self.term_to_idx = {term: idx for idx, term in enumerate(self.vocabulary)}
        
        # Calculate document frequencies for IDF
        doc_frequencies = Counter()
        for term_freq in doc_term_frequencies:
            for term in term_freq.keys():
                doc_frequencies[term] += 1
        
        self.doc_frequencies = doc_frequencies
        self.doc_term_frequencies = doc_term_frequencies
        self.is_trained = True
        
        end_time = time.time()
        
        return {
            'indexing_time': end_time - start_time,
            'total_documents': len(chunks),
            'vocabulary_size': self.vocab_size,
            'avg_doc_length': self.avg_doc_length
        }
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document"""
        score = 0.0
        doc_length = self.doc_lengths[doc_idx]
        doc_term_freq = self.doc_term_frequencies[doc_idx]
        
        for term in query_terms:
            if term in doc_term_freq:
                # Term frequency in document
                tf = doc_term_freq[term]
                
                # Document frequency (for IDF calculation)
                df = self.doc_frequencies.get(term, 0)
                
                # IDF calculation
                idf = math.log((len(self.chunks) - df + 0.5) / (df + 0.5))
                
                # BM25 score component
                numerator = idf * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                
                score += numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Search for most relevant chunks using BM25"""
        if not self.is_trained:
            raise ValueError("Index not built. Call build_index() first.")
        
        start_time = time.time()
        
        # Preprocess and tokenize query
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return [], {'search_time': 0, 'method': 'bm25', 'results_count': 0}
        
        # Calculate BM25 scores for all documents
        scores = []
        for doc_idx in range(len(self.chunks)):
            score = self._calculate_bm25_score(query_terms, doc_idx)
            scores.append((score, doc_idx))
        
        # Sort by score (descending)
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Prepare results
        results = []
        for rank, (score, doc_idx) in enumerate(scores[:top_k]):
            if score > 0:  # Only include documents with positive scores
                chunk = self.chunks[doc_idx].copy()
                chunk['retrieval_rank'] = rank + 1
                chunk['retrieval_score'] = float(score)
                chunk['retrieval_method'] = 'bm25'
                results.append(chunk)
        
        end_time = time.time()
        
        metadata = {
            'search_time': end_time - start_time,
            'query_terms': query_terms,
            'results_count': len(results),
            'method': 'bm25'
        }
        
        return results, metadata


class HybridRetriever:
    """Hybrid retrieval combining vector and BM25 methods"""
    
    def __init__(self, vector_weight: float = 0.7, bm25_weight: float = 0.3):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.vector_retriever = None
        self.bm25_retriever = None
        self.is_trained = False
    
    def build_index(self, chunks: List[Dict[str, Any]], dimension: int) -> Dict[str, Any]:
        """Build both vector and BM25 indices"""
        start_time = time.time()
        
        # Get actual dimension from first chunk if available
        if chunks and 'embedding' in chunks[0]:
            actual_dimension = len(chunks[0]['embedding'])
            dimension = actual_dimension
        
        # Initialize retrievers
        self.vector_retriever = VectorRetriever(dimension)
        self.bm25_retriever = BM25Retriever()
        
        # Build indices
        vector_metadata = self.vector_retriever.build_index(chunks)
        bm25_metadata = self.bm25_retriever.build_index(chunks)
        
        self.is_trained = True
        
        end_time = time.time()
        
        return {
            'total_indexing_time': end_time - start_time,
            'vector_indexing_time': vector_metadata['indexing_time'],
            'bm25_indexing_time': bm25_metadata['indexing_time'],
            'total_documents': len(chunks),
            'vector_dimension': dimension,
            'bm25_vocabulary_size': bm25_metadata['vocabulary_size'],
            'weights': {
                'vector': self.vector_weight,
                'bm25': self.bm25_weight
            }
        }
    
    def search(self, query: str, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Hybrid search combining vector and BM25 results"""
        if not self.is_trained:
            raise ValueError("Index not built. Call build_index() first.")
        
        start_time = time.time()
        
        # Get results from both methods
        vector_results, vector_metadata = self.vector_retriever.search(query_embedding, top_k * 2)
        bm25_results, bm25_metadata = self.bm25_retriever.search(query, top_k * 2)
        
        # Normalize scores to [0, 1] range
        def normalize_scores(results: List[Dict[str, Any]], score_key: str = 'retrieval_score'):
            if not results:
                return results
            
            scores = [r[score_key] for r in results]
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                for r in results:
                    r[f'normalized_{score_key}'] = 1.0
            else:
                for r in results:
                    r[f'normalized_{score_key}'] = (r[score_key] - min_score) / (max_score - min_score)
            
            return results
        
        # Normalize scores
        vector_results = normalize_scores(vector_results)
        bm25_results = normalize_scores(bm25_results)
        
        # Create combined scores
        combined_scores = defaultdict(float)
        chunk_data = {}
        
        # Add vector scores
        for result in vector_results:
            chunk_id = result['id']
            combined_scores[chunk_id] += self.vector_weight * result['normalized_retrieval_score']
            chunk_data[chunk_id] = result
            chunk_data[chunk_id]['vector_score'] = result['retrieval_score']
            chunk_data[chunk_id]['vector_rank'] = result['retrieval_rank']
        
        # Add BM25 scores
        for result in bm25_results:
            chunk_id = result['id']
            combined_scores[chunk_id] += self.bm25_weight * result['normalized_retrieval_score']
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result
            chunk_data[chunk_id]['bm25_score'] = result['retrieval_score']
            chunk_data[chunk_id]['bm25_rank'] = result['retrieval_rank']
        
        # Sort by combined score
        sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare final results
        results = []
        for rank, (chunk_id, combined_score) in enumerate(sorted_chunks[:top_k]):
            chunk = chunk_data[chunk_id].copy()
            chunk['retrieval_rank'] = rank + 1
            chunk['retrieval_score'] = combined_score
            chunk['retrieval_method'] = 'hybrid'
            
            # Add individual method scores if available
            if 'vector_score' not in chunk:
                chunk['vector_score'] = 0.0
                chunk['vector_rank'] = None
            if 'bm25_score' not in chunk:
                chunk['bm25_score'] = 0.0
                chunk['bm25_rank'] = None
            
            results.append(chunk)
        
        end_time = time.time()
        
        metadata = {
            'search_time': end_time - start_time,
            'vector_search_time': vector_metadata['search_time'],
            'bm25_search_time': bm25_metadata['search_time'],
            'results_count': len(results),
            'method': 'hybrid',
            'weights': {
                'vector': self.vector_weight,
                'bm25': self.bm25_weight
            },
            'component_results': {
                'vector_count': len(vector_results),
                'bm25_count': len(bm25_results)
            }
        }
        
        return results, metadata


class RetrievalSystem:
    """Main retrieval system orchestrator"""
    
    def __init__(self):
        self.retrievers = {}
        self.indexed_configs = {}
    
    def build_index(self, chunks: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Build retrieval index based on configuration"""
        retriever_type = config['retriever']
        config_key = self._get_config_key(config)
        
        start_time = time.time()
        
        if retriever_type == 'vector':
            dimension = config['embedding']['dimension']
            retriever = VectorRetriever(dimension)
            index_metadata = retriever.build_index(chunks)
            
        elif retriever_type == 'bm25':
            retriever = BM25Retriever()
            index_metadata = retriever.build_index(chunks)
            
        elif retriever_type == 'hybrid':
            dimension = config['embedding']['dimension']
            hybrid_alpha = config.get('hybrid_alpha', 0.7)
            retriever = HybridRetriever(vector_weight=hybrid_alpha, bm25_weight=1.0-hybrid_alpha)
            index_metadata = retriever.build_index(chunks, dimension)
            
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        
        # Store retriever
        self.retrievers[config_key] = retriever
        self.indexed_configs[config_key] = config
        
        end_time = time.time()
        
        return {
            **index_metadata,
            'config_key': config_key,
            'retriever_type': retriever_type,
            'total_setup_time': end_time - start_time
        }
    
    def search(self, query: str, query_embedding: np.ndarray, config: Dict[str, Any], top_k: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Perform search using specified configuration"""
        config_key = self._get_config_key(config)
        retriever = self.retrievers.get(config_key)
        
        if not retriever:
            raise ValueError(f"No retriever found for config: {config_key}")
        
        retriever_type = config['retriever']
        
        if retriever_type == 'vector':
            return retriever.search(query_embedding, top_k)
        elif retriever_type == 'bm25':
            return retriever.search(query, top_k)
        elif retriever_type == 'hybrid':
            return retriever.search(query, query_embedding, top_k)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    def _get_config_key(self, config: Dict[str, Any]) -> str:
        """Generate unique key for configuration"""
        embedding = config['embedding']
        chunking = config['chunking']
        retriever = config['retriever']
        
        hybrid_suffix = f"_a{config.get('hybrid_alpha', 0.7)}" if retriever == 'hybrid' else ""
        return f"{embedding['provider']}_{embedding['dimension']}_{chunking['type']}_{chunking['params']}_{retriever}{hybrid_suffix}"
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """Get information about available retrievers"""
        return {
            'total_retrievers': len(self.retrievers),
            'retriever_types': list(set(
                config['retriever'] for config in self.indexed_configs.values()
            )),
            'configurations': list(self.indexed_configs.keys())
        }


def test_retrieval_system():
    """Test function for retrieval system"""
    # Test data
    test_chunks = [
        {
            'id': 0,
            'text': 'Machine learning is a subset of artificial intelligence.',
            'embedding': np.random.rand(384).astype(np.float32)
        },
        {
            'id': 1,
            'text': 'Natural language processing helps computers understand human language.',
            'embedding': np.random.rand(384).astype(np.float32)
        },
        {
            'id': 2,
            'text': 'Retrieval augmented generation combines search with language models.',
            'embedding': np.random.rand(384).astype(np.float32)
        }
    ]
    
    test_config = {
        'embedding': {'provider': 'test', 'dimension': 384},
        'chunking': {'type': 'fixed', 'params': '500c-50o'},
        'retriever': 'hybrid'
    }
    
    try:
        system = RetrievalSystem()
        
        # Build index
        index_result = system.build_index(test_chunks, test_config)
        print(f"Index built in {index_result['total_setup_time']:.3f}s")
        
        # Test search
        query = "What is machine learning?"
        query_embedding = np.random.rand(384).astype(np.float32)
        
        results, metadata = system.search(query, query_embedding, test_config, top_k=2)
        print(f"Found {len(results)} results in {metadata['search_time']:.3f}s")
        
        for result in results:
            print(f"Rank {result['retrieval_rank']}: Score {result['retrieval_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_retrieval_system()
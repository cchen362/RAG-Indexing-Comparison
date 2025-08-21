"""
Main RAG pipeline orchestrator for RAG Indexing Comparison App
Integrates document processing, embedding, retrieval, and response generation
"""

import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple
import openai
import streamlit as st
from dotenv import load_dotenv
import os

from document_processor import DocumentProcessor, DocumentChunker
from embedding_pipeline import EmbeddingPipeline
from retrieval_system import RetrievalSystem
from sheets_logger import SheetsLogger
from logger_config import get_logger

# Load environment variables
load_dotenv()

class RAGPipeline:
    """Main RAG pipeline orchestrator"""
    
    def __init__(self):
        self.logger = get_logger('rag_pipeline')
        self.logger.info("🔧 Initializing RAG Pipeline components...")
        
        self.document_processor = DocumentProcessor()
        self.document_chunker = DocumentChunker()
        self.embedding_pipeline = EmbeddingPipeline()
        self.retrieval_system = RetrievalSystem()
        self.sheets_logger = SheetsLogger()
        
        # Initialize OpenAI client for generation
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.openai_client = openai.OpenAI(api_key=api_key)
        else:
            self.openai_client = None
            st.warning("OpenAI API key not found. Response generation will be disabled.")
    
    def process_documents(self, uploaded_files) -> List[Dict[str, Any]]:
        """Process uploaded documents and extract text"""
        return self.document_processor.process_uploaded_files(uploaded_files)
    
    def execute_pipeline(self, query: str, documents: List[Dict[str, Any]], configs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Execute the complete RAG pipeline for all configurations"""
        results = []
        
        for config in configs:
            try:
                result = self._execute_single_config(query, documents, config, top_k)
                results.append(result)
                
                # Log to Google Sheets in background
                try:
                    self.sheets_logger.log_pipeline_result(result)
                except Exception as e:
                    st.warning(f"Failed to log result to Google Sheets: {str(e)}")
                
            except Exception as e:
                st.error(f"Pipeline failed for config {config['name']}: {str(e)}")
                
                # Create error result
                error_result = {
                    'config': config,
                    'query': query,
                    'run_id': str(uuid.uuid4())[:8],
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'success': False
                }
                results.append(error_result)
        
        return results
    
    def _execute_single_config(self, query: str, documents: List[Dict[str, Any]], config: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        """Execute pipeline for a single configuration"""
        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        self.logger.info(f"📋 Executing config: {config['name']} (run_id: {run_id})")
        self.logger.debug(f"📋 Query: {query}")
        self.logger.debug(f"📋 Documents: {len(documents)} files")
        
        timing = {}
        
        # Step 1: Document Chunking
        chunking_start = time.time()
        self.logger.debug(f"🔪 Starting chunking with strategy: {config['chunking']['type']}")
        chunks = self.document_chunker.chunk_documents(documents, config['chunking'])
        timing['chunking_time'] = time.time() - chunking_start
        self.logger.info(f"🔪 Chunking completed: {len(chunks)} chunks in {timing['chunking_time']:.2f}s")
        
        if not chunks:
            raise ValueError("No chunks created from documents")
        
        # Step 2: Create Embeddings
        embedding_start = time.time()
        self.logger.debug(f"🔢 Starting embeddings with: {config['embedding']['provider']}")
        embedding_result = self.embedding_pipeline.create_embeddings(chunks, config)
        enriched_chunks = embedding_result['chunks']
        timing['embedding_time'] = embedding_result['metadata']['total_processing_time']
        self.logger.info(f"🔢 Embeddings completed: {len(enriched_chunks)} chunks in {timing['embedding_time']:.2f}s")
        
        # Step 3: Build Retrieval Index
        indexing_start = time.time()
        index_metadata = self.retrieval_system.build_index(enriched_chunks, config)
        timing['indexing_time'] = time.time() - indexing_start
        
        # Step 4: Create Query Embedding
        query_embedding_start = time.time()
        query_result = self.embedding_pipeline.embed_query(query, config)
        query_embedding = query_result['embedding']
        timing['query_embedding_time'] = time.time() - query_embedding_start
        
        # Step 5: Retrieve Relevant Chunks
        retrieval_start = time.time()
        retrieved_chunks, retrieval_metadata = self.retrieval_system.search(
            query, query_embedding, config, top_k
        )
        retrieval_end = time.time()
        timing['retrieval_time'] = retrieval_end - retrieval_start
        timing['retrieval_latency_ms'] = (retrieval_end - retrieval_start) * 1000  # Convert to milliseconds
        
        # Step 6: Generate Response
        generation_start = time.time()
        response, generation_tokens = self._generate_response(query, retrieved_chunks)
        timing['generation_time'] = time.time() - generation_start
        
        # Calculate total time
        timing['total_time'] = time.time() - start_time
        
        # Extract token counts
        embedding_tokens = query_result['metadata'].get('total_tokens', 0)
        retrieval_candidates = len(retrieved_chunks)
        
        # Prepare result
        result = {
            'config': config,
            'query': query,
            'run_id': run_id,
            'timestamp': datetime.now(),
            'timing': timing,
            'response': response,
            'retrieved_chunks': retrieved_chunks,
            'document_count': len(documents),
            'total_chunks': len(chunks),
            'top_k': top_k,
            'embedding_tokens': embedding_tokens,
            'retrieval_candidates': retrieval_candidates,
            'generation_input_tokens': generation_tokens.get('input_tokens', 0),
            'generation_output_tokens': generation_tokens.get('output_tokens', 0),
            'success': True,
            'metadata': {
                'embedding_metadata': embedding_result['metadata'],
                'index_metadata': index_metadata,
                'retrieval_metadata': retrieval_metadata,
                'query_metadata': query_result['metadata']
            }
        }
        
        return result
    
    def _generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> tuple:
        """Generate response using GPT-4o-mini based on retrieved chunks"""
        if not self.openai_client:
            return "Response generation unavailable (OpenAI API key not configured)", {'input_tokens': 0, 'output_tokens': 0}
        
        if not retrieved_chunks:
            return "No relevant information found in the documents to answer your question.", {'input_tokens': 0, 'output_tokens': 0}
        
        try:
            # Prepare context from retrieved chunks
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks[:5], 1):  # Use top 5 chunks
                doc_id = chunk.get('document_id', 'Unknown Document')
                text = chunk.get('text', '')
                score = chunk.get('retrieval_score', 0)
                context_parts.append(f"[Source {i} - {doc_id} (Score: {score:.3f})]:\n{text}")
            
            context = "\n\n".join(context_parts)
            
            # Create prompt
            prompt = f"""Based on the following context from retrieved documents, please answer the user's question. Be concise and accurate. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context. Be accurate and cite relevant sources when possible."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            # Extract token usage
            usage = response.usage
            token_info = {
                'input_tokens': usage.prompt_tokens,
                'output_tokens': usage.completion_tokens
            }
            
            return response.choices[0].message.content.strip(), token_info
            
        except Exception as e:
            return f"Error generating response: {str(e)}", {'input_tokens': 0, 'output_tokens': 0}
    
    def get_pipeline_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate pipeline performance statistics"""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'total_runs': len(results), 'successful_runs': 0, 'success_rate': 0}
        
        # Calculate timing statistics
        total_times = [r['timing']['total_time'] for r in successful_results]
        embedding_times = [r['timing']['embedding_time'] for r in successful_results]
        retrieval_times = [r['timing']['retrieval_time'] for r in successful_results]
        generation_times = [r['timing']['generation_time'] for r in successful_results]
        
        stats = {
            'total_runs': len(results),
            'successful_runs': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'avg_total_time': sum(total_times) / len(total_times),
            'avg_embedding_time': sum(embedding_times) / len(embedding_times),
            'avg_retrieval_time': sum(retrieval_times) / len(retrieval_times),
            'avg_generation_time': sum(generation_times) / len(generation_times),
            'fastest_config': min(successful_results, key=lambda x: x['timing']['total_time'])['config']['name'],
            'slowest_config': max(successful_results, key=lambda x: x['timing']['total_time'])['config']['name']
        }
        
        return stats
    
    def validate_setup(self) -> Dict[str, bool]:
        """Validate that all components are properly set up"""
        validation_results = {
            'document_processor': True,  # Always available
            'embedding_pipeline': False,
            'retrieval_system': True,    # Always available  
            'sheets_logger': False,
            'openai_generation': bool(self.openai_client)
        }
        
        # Test embedding pipeline
        try:
            available_models = self.embedding_pipeline.get_available_models()
            validation_results['embedding_pipeline'] = len(available_models) > 0
        except Exception:
            pass
        
        # Test sheets logger
        try:
            validation_results['sheets_logger'] = self.sheets_logger.test_connection()
        except Exception:
            pass
        
        return validation_results


class PipelineProgress:
    """Helper class for tracking pipeline progress in Streamlit"""
    
    def __init__(self, total_configs: int):
        self.total_configs = total_configs
        self.current_config = 0
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.current_step_text = st.empty()
        
    def update_config(self, config_name: str):
        """Update progress for new configuration"""
        self.current_config += 1
        progress = self.current_config / self.total_configs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Processing configuration {self.current_config}/{self.total_configs}: {config_name}")
    
    def update_step(self, step_name: str):
        """Update current step within configuration"""
        self.current_step_text.text(f"   → {step_name}")
    
    def complete(self):
        """Mark progress as complete"""
        self.progress_bar.progress(1.0)
        self.status_text.text("✅ All configurations completed!")
        self.current_step_text.text("")


def test_rag_pipeline():
    """Test function for RAG pipeline"""
    pipeline = RAGPipeline()
    
    # Validate setup
    setup_status = pipeline.validate_setup()
    print("Setup validation:")
    for component, status in setup_status.items():
        print(f"  {component}: {'✅' if status else '❌'}")
    
    if not setup_status['embedding_pipeline']:
        print("❌ Embedding pipeline not available")
        return False
    
    # Test with mock data (would normally use uploaded files)
    test_documents = [
        {
            'filename': 'test.txt',
            'content': 'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
            'size': 100,
            'file_type': 'txt'
        }
    ]
    
    test_config = {
        'name': 'test_config',
        'embedding': {
            'provider': 'openai',
            'model': 'text-embedding-3-small',
            'dimension': 1536
        },
        'chunking': {
            'type': 'fixed',
            'params': '200c-20o'
        },
        'retriever': 'vector'
    }
    
    try:
        results = pipeline.execute_pipeline(
            query="What is machine learning?",
            documents=test_documents,
            configs=[test_config],
            top_k=3
        )
        
        if results and results[0].get('success'):
            print("✅ Pipeline test successful")
            print(f"   Response: {results[0]['response'][:100]}...")
            return True
        else:
            print("❌ Pipeline test failed")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_rag_pipeline()
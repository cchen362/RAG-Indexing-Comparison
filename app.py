#!/usr/bin/env python3
"""
RAG Indexing Comparison App
A Streamlit app for testing and comparing different RAG pipeline configurations
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import time
import uuid
from datetime import datetime

# Import RAG pipeline components
from rag_pipeline import RAGPipeline

# Configure page with Amex GBT-inspired theme
st.set_page_config(
    page_title="RAG Pipeline Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SOPHISTICATED AMEX STYLING - REBUILT SAFELY
def load_amex_styling():
    """Sophisticated Amex GBT styling - rebuilt with surgical precision"""
    st.markdown("""
    <style>
    /* Amex GBT Professional Color System */
    :root {
        --amex-deep-blue: #00175A;
        --amex-bright-blue: #006FCF;
        --amex-charcoal: #152835;
        --amex-light-gray: #F8F9FA;
        --amex-powder-blue: #EDF4FB;
        --amex-white: #FFFFFF;
        --amex-accent-gold: #FFD700;
        --amex-success: #28A745;
    }
    
    /* SAFE: App Background */
    .stApp {
        background-color: var(--amex-light-gray);
    }
    
    /* SAFE: Main Content Container */
    .main .block-container {
        background: var(--amex-white);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 23, 90, 0.08);
        padding: 2.5rem;
        margin: 1.5rem;
        border: 1px solid #E9ECEF;
    }
    
    /* SAFE: Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--amex-deep-blue) 0%, var(--amex-charcoal) 100%);
        border-right: 3px solid var(--amex-bright-blue);
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 1rem;
    }
    
    /* SAFE: Sidebar Typography */
    section[data-testid="stSidebar"] * {
        color: white;
    }
    
    section[data-testid="stSidebar"] h1 {
        color: var(--amex-accent-gold);
        text-align: center;
        border-bottom: 2px solid var(--amex-accent-gold);
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    
    section[data-testid="stSidebar"] h3 {
        color: #B3D9FF;
        border-bottom: 1px solid var(--amex-bright-blue);
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 0.75rem 0;
    }
    
    /* SAFE: Sidebar Description Boxes */
    section[data-testid="stSidebar"] em {
        color: #E3F2FD;
        display: block;
        margin: 0.5rem 0 1rem 0;
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        border-left: 3px solid #4FC3F7;
        line-height: 1.4;
    }
    
    /* SAFE: Sidebar Controls */
    section[data-testid="stSidebar"] .stCheckbox {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        padding: 0.6rem;
        margin: 0.4rem 0;
    }
    
    section[data-testid="stSidebar"] .stSlider {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 6px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    
    /* SAFE: Main Content Headers */
    .main h1 {
        color: var(--amex-deep-blue);
        text-align: center;
    }
    
    .main h2 {
        color: var(--amex-deep-blue);
        border-bottom: 2px solid var(--amex-bright-blue);
        padding-bottom: 0.5rem;
    }
    
    /* SAFE: Professional Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--amex-bright-blue) 0%, var(--amex-deep-blue) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 111, 207, 0.25);
    }
    
    /* SAFE: File Uploader */
    .stFileUploader {
        background: var(--amex-powder-blue);
        border: 2px dashed var(--amex-bright-blue);
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* SAFE: Result Cards */
    .stExpander {
        border: 1px solid #E9ECEF;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 23, 90, 0.08);
    }
    
    /* SAFE: Progress Bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--amex-bright-blue) 0%, var(--amex-deep-blue) 100%);
        height: 8px;
        border-radius: 4px;
    }
    
    /* SAFE: Tooltip Enhancement */
    .stTooltipIcon {
        color: var(--amex-accent-gold);
        font-size: 1rem;
    }
    
    /* SAFE: Success Messages */
    .stSuccess {
        background: linear-gradient(135deg, var(--amex-success) 0%, #20c997 100%);
        border-radius: 8px;
        color: white;
    }
    
    /* SAFE: Info Messages */
    .stInfo {
        background: var(--amex-powder-blue);
        border-left: 4px solid var(--amex-bright-blue);
        border-radius: 8px;
        color: var(--amex-deep-blue);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* SAFE: Subtle Animation */
    .main .block-container {
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Import logging
from logger_config import setup_logging, get_logger

# Setup logging
setup_logging(log_to_file=True)
logger = get_logger('streamlit_app')

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Indexing Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .comparison-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin: 0.5rem 0;
    }
    .config-group {
        background-color: #f1f3f4;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'run_id' not in st.session_state:
        st.session_state.run_id = str(uuid.uuid4())[:8]
        logger.info(f"🆔 New session initialized with run_id: {st.session_state.run_id}")
    if 'uploaded_docs' not in st.session_state:
        st.session_state.uploaded_docs = []
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = []

def render_sidebar():
    """Render the sidebar with configuration options"""
    with st.sidebar:
        st.title("🔍 RAG Configuration")
        
        # Embedding Models Section
        st.markdown("### 📊 Embedding Models")
        st.markdown("*How the AI 'reads' and understands your documents - like choosing between different types of reading specialists*")
        with st.container():
            st.markdown('<div class="config-group">', unsafe_allow_html=True)
            
            # OpenAI Embeddings
            openai_small = st.checkbox("OpenAI text-embedding-3-small (1536-dim)", value=True, key="openai_small", 
                                     help="Like a smart librarian who reads documents quickly and creates compact summaries for fast searching")
            openai_large = st.checkbox("OpenAI text-embedding-3-large (3072-dim)", value=False, key="openai_large",
                                     help="Like a detailed scholar who creates comprehensive summaries - slower but captures more nuance")
            
            # Cohere Embeddings
            cohere_default = st.checkbox("Cohere embed-v4.0 (auto-dim)", value=True, key="cohere_default",
                                       help="Like a specialist translator who excels at understanding context and relationships between ideas")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chunking Strategy Section
        st.markdown("### 📝 Chunking Strategy")
        st.markdown("*How we break your documents into digestible pieces - like choosing how to slice a pizza for optimal sharing*")
        with st.container():
            st.markdown('<div class="config-group">', unsafe_allow_html=True)
            
            # Fixed-size chunking
            fixed_chunking = st.checkbox("Fixed-size chunking", value=True, key="fixed_chunking",
                                       help="Like cutting a book into equal-sized chapters - consistent and predictable")
            if fixed_chunking:
                chunk_size = st.slider("Chunk size (characters)", 200, 1000, 500, 50, key="chunk_size",
                                     help="How many characters in each piece - like setting page length")
                chunk_overlap = st.slider("Chunk overlap", 0, 200, 50, 10, key="chunk_overlap",
                                        help="How much content to repeat between pieces - like having chapter previews")
            
            # Sentence-based chunking
            sentence_chunking = st.checkbox("Sentence-based chunking", value=False, key="sentence_chunking",
                                          help="Like a smart editor who breaks content at natural topic boundaries for better understanding")
            if sentence_chunking:
                max_sentences = st.slider("Max sentences per chunk", 3, 10, 5, 1, key="max_sentences",
                                        help="Maximum sentences per piece - like setting paragraph length")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Retrieval Method Section
        st.markdown("### 🎯 Retrieval Method")
        st.markdown("*How the system finds the most relevant information to answer your question - like choosing your search strategy in a library*")
        with st.container():
            st.markdown('<div class="config-group">', unsafe_allow_html=True)
            
            vector_only = st.checkbox("Vector-only (cosine similarity)", value=True, key="vector_only",
                                    help="Like having a personal assistant who finds documents based on meaning and context, not just keywords")
            bm25_only = st.checkbox("BM25-only (keyword-based)", value=False, key="bm25_only",
                                  help="Like a traditional search engine that matches your exact words and phrases")
            hybrid_30 = st.checkbox("Hybrid (30% vector + 70% BM25)", value=False, key="hybrid_30",
                                  help="Like combining a librarian's intuition with a search engine's precision - balanced toward keyword matching")
            hybrid_50 = st.checkbox("Hybrid (50% vector + 50% BM25)", value=False, key="hybrid_50",
                                  help="Like combining a librarian's intuition with a search engine's precision - perfectly balanced")
            hybrid_70 = st.checkbox("Hybrid (70% vector + 30% BM25)", value=True, key="hybrid_70",
                                  help="Like combining a librarian's intuition with a search engine's precision - balanced toward meaning understanding")
            
            # Top-k results
            top_k = st.slider("Top-k results to retrieve", 1, 20, 5, 1, key="top_k",
                            help="How many of the most relevant documents to consider - like asking for the 'top 5' recommendations")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Generation Model (Fixed)
        st.markdown("### 🤖 Generation Model")
        st.markdown("*The AI writer that crafts the final response using the information found from your documents*")
        st.info("**Fixed:** GPT-4o-mini (for consistent comparison)")
        
        # Configuration Summary
        st.markdown("### ⚙️ Active Configurations")
        active_configs = get_active_configurations()
        if active_configs:
            st.success(f"**{len(active_configs)} pipeline(s)** will run")
            for i, config in enumerate(active_configs, 1):
                st.caption(f"{i}. {config['name']}")
        else:
            st.warning("No configurations selected")

def get_active_configurations():
    """Get list of active pipeline configurations based on sidebar selections"""
    configs = []
    
    # Get selected embeddings
    embeddings = []
    if st.session_state.get('openai_small', False):
        embeddings.append(('openai', 'text-embedding-3-small', 1536))
    if st.session_state.get('openai_large', False):
        embeddings.append(('openai', 'text-embedding-3-large', 3072))
    if st.session_state.get('cohere_default', False):
        embeddings.append(('cohere', 'embed-v4.0', 1536))  # Will be auto-detected
    
    # Get selected chunking strategies
    chunking_strategies = []
    if st.session_state.get('fixed_chunking', False):
        chunk_size = st.session_state.get('chunk_size', 500)
        chunk_overlap = st.session_state.get('chunk_overlap', 50)
        chunking_strategies.append(('fixed', f"{chunk_size}c-{chunk_overlap}o"))
    if st.session_state.get('sentence_chunking', False):
        max_sentences = st.session_state.get('max_sentences', 5)
        chunking_strategies.append(('sentence', f"{max_sentences}s"))
    
    # Get selected retrieval methods with their specific alpha values
    retrievers = []
    if st.session_state.get('vector_only', False):
        retrievers.append(('vector', None))
    if st.session_state.get('bm25_only', False):
        retrievers.append(('bm25', None))
    if st.session_state.get('hybrid_30', False):
        retrievers.append(('hybrid', 0.3))
    if st.session_state.get('hybrid_50', False):
        retrievers.append(('hybrid', 0.5))
    if st.session_state.get('hybrid_70', False):
        retrievers.append(('hybrid', 0.7))
    
    # Generate all combinations
    for embed_provider, embed_model, embed_dim in embeddings:
        for chunk_type, chunk_params in chunking_strategies:
            for retriever, alpha in retrievers:
                if retriever == 'hybrid' and alpha is not None:
                    config_name = f"{embed_provider}-{embed_dim}d_{chunk_type}-{chunk_params}_{retriever}_a{alpha}"
                    configs.append({
                        'name': config_name,
                        'embedding': {
                            'provider': embed_provider,
                            'model': embed_model, 
                            'dimension': embed_dim
                        },
                        'chunking': {
                            'type': chunk_type,
                            'params': chunk_params
                        },
                        'retriever': retriever,
                        'hybrid_alpha': alpha
                    })
                else:
                    # Regular config for non-hybrid retrievers
                    config_name = f"{embed_provider}-{embed_dim}d_{chunk_type}-{chunk_params}_{retriever}"
                    configs.append({
                        'name': config_name,
                        'embedding': {
                            'provider': embed_provider,
                            'model': embed_model, 
                            'dimension': embed_dim
                        },
                        'chunking': {
                            'type': chunk_type,
                            'params': chunk_params
                        },
                        'retriever': retriever
                    })
    
    return configs

def render_document_upload():
    """Render document upload section"""
    st.markdown("## 📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload documents for indexing",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT (max 200MB per file)"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
        logger.info(f"📄 {len(uploaded_files)} files uploaded: {[f.name for f in uploaded_files]}")
        
        # Display uploaded files
        with st.expander("📋 Uploaded Files", expanded=False):
            for file in uploaded_files:
                file_size = len(file.getvalue()) / (1024 * 1024)  # MB
                st.write(f"• **{file.name}** ({file_size:.2f} MB)")
                logger.debug(f"📄 File: {file.name} ({file_size:.2f} MB)")
        
        # Store in session state
        st.session_state.uploaded_docs = uploaded_files
        return True
    
    return False

def render_query_section():
    """Render query input and execution section"""
    st.markdown("## 🔍 Query & Execution")
    
    # Query input
    query = st.text_area(
        "Enter your query",
        placeholder="e.g., What is retrieval-augmented generation?",
        height=100,
        key="user_query"
    )
    
    # Execution controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        run_comparison = st.button(
            "🚀 Run Comparison", 
            type="primary",
            disabled=not (query and st.session_state.uploaded_docs and get_active_configurations())
        )
    
    with col2:
        parallel_runs = st.selectbox("Parallel runs", [1, 2, 3], index=1, key="parallel_runs")
    
    with col3:
        if st.button("🔄 Reset Results"):
            st.session_state.pipeline_results = []
            st.rerun()
    
    # Validation messages
    if run_comparison:
        logger.info(f"🚀 Run comparison button clicked for query: '{query[:50]}...'")
        if not query:
            st.error("Please enter a query")
            logger.warning("❌ No query provided")
        elif not st.session_state.uploaded_docs:
            st.error("Please upload documents first")
            logger.warning("❌ No documents uploaded")
        elif not get_active_configurations():
            st.error("Please select at least one configuration")
            logger.warning("❌ No configurations selected")
        else:
            # Generate new Run ID for this comparison
            st.session_state.run_id = str(uuid.uuid4())[:8]
            logger.info(f"🆔 New comparison started with run_id: {st.session_state.run_id}")
            
            # Execute pipeline comparison
            configs = get_active_configurations()
            logger.info(f"🔧 Executing {len(configs)} configurations: {[c['name'] for c in configs]}")
            execute_pipeline_comparison(query, parallel_runs)
    
    return query, run_comparison

def execute_pipeline_comparison(query, parallel_runs):
    """Execute pipeline comparison with selected configurations"""
    configs = get_active_configurations()
    documents = st.session_state.uploaded_docs
    
    st.markdown("## ⚡ Pipeline Execution")
    
    # Initialize pipeline
    if 'rag_pipeline' not in st.session_state:
        try:
            logger.info("🔄 Initializing RAG pipeline...")
            st.session_state.rag_pipeline = RAGPipeline()
            st.success("✅ Pipeline initialized successfully")
            logger.info("✅ Pipeline initialized successfully")
        except Exception as e:
            st.error(f"❌ Failed to initialize pipeline: {str(e)}")
            logger.error(f"❌ Pipeline initialization failed: {str(e)}")
            return
    
    pipeline = st.session_state.rag_pipeline
    
    # Validate setup
    setup_status = pipeline.validate_setup()
    if not setup_status['embedding_pipeline']:
        st.error("❌ Embedding pipeline not available. Please check your API keys.")
        return
    
    # Process documents first
    with st.spinner("Processing uploaded documents..."):
        processed_docs = pipeline.process_documents(documents)
        if not processed_docs:
            st.error("No documents could be processed.")
            return
    
    # Progress tracking
    from rag_pipeline import PipelineProgress
    progress_tracker = PipelineProgress(len(configs))
    
    # Execute pipeline for all configurations
    try:
        with st.spinner("Executing RAG pipeline..."):
            # Get top_k from session state
            top_k = st.session_state.get('top_k', 5)
            
            # Execute pipeline
            results = []
            for config in configs:
                progress_tracker.update_config(config['name'])
                
                # Execute single configuration
                try:
                    logger.info(f"⚡ Starting execution for config: {config['name']}")
                    progress_tracker.update_step("Processing documents...")
                    single_result = pipeline._execute_single_config(query, processed_docs, config, top_k, st.session_state.run_id)
                    results.append(single_result)
                    
                    # Log success
                    timing = single_result.get('timing', {})
                    logger.info(f"✅ Config {config['name']} completed in {timing.get('total_time', 0):.2f}s")
                    
                    # Log to sheets in background
                    try:
                        pipeline.sheets_logger.log_pipeline_result(single_result)
                        logger.debug(f"📊 Results logged to Google Sheets for {config['name']}")
                    except Exception as e:
                        st.warning(f"Failed to log to Google Sheets: {str(e)}")
                        logger.warning(f"📊 Failed to log to Google Sheets: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Failed to execute config {config['name']}: {str(e)}")
                    st.error(f"Error details: {type(e).__name__}: {str(e)}")
                    logger.error(f"❌ Config {config['name']} failed: {type(e).__name__}: {str(e)}")
                    error_result = {
                        'config': config,
                        'query': query,
                        'run_id': st.session_state.run_id,
                        'timestamp': datetime.now(),
                        'error': str(e),
                        'success': False,
                        'timing': {'total_time': 0}  # Add empty timing for consistency
                    }
                    results.append(error_result)
            
            progress_tracker.complete()
            
            # Store results
            st.session_state.pipeline_results = results
            
            # Display summary
            successful_results = [r for r in results if r.get('success', False)]
            if successful_results:
                st.success(f"✅ Successfully executed {len(successful_results)}/{len(configs)} configurations")
                
                # Show quick stats
                stats = pipeline.get_pipeline_stats(results)
                col1, col2, col3 = st.columns(3)
                col1.metric("Success Rate", f"{stats['success_rate']:.1%}")
                col2.metric("Avg Total Time", f"{stats['avg_total_time']:.2f}s")
                col3.metric("Fastest Config", stats['fastest_config'].split('_')[0])
            else:
                st.error("❌ All configurations failed to execute")
                
    except Exception as e:
        st.error(f"Pipeline execution failed: {str(e)}")
        return

def render_results():
    """Render comparison results"""
    if not st.session_state.pipeline_results:
        return
    
    st.markdown("## 📊 Comparison Results")
    
    # Results overview
    total_results = len(st.session_state.pipeline_results)
    successful_results = [r for r in st.session_state.pipeline_results if r.get('success', False) and 'timing' in r]
    avg_time = sum(r['timing']['total_time'] for r in successful_results) / len(successful_results) if successful_results else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Runs", total_results)
    col2.metric("Average Time", f"{avg_time:.2f}s")
    col3.metric("Run ID", st.session_state.run_id)
    
    # Individual results
    for i, result in enumerate(st.session_state.pipeline_results):
        with st.expander(f"🔸 {result['config']['name']}", expanded=i < 2):
            render_individual_result(result)

def render_individual_result(result):
    """Render individual pipeline result"""
    # Handle error results
    if not result.get('success', True):
        st.error(f"❌ Configuration failed: {result.get('error', 'Unknown error')}")
        return
    
    config = result['config']
    timing = result.get('timing', {})
    
    # Configuration details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Configuration:**")
        st.write(f"• Embedding: {config['embedding']['provider']} ({config['embedding']['dimension']}d)")
        st.write(f"• Chunking: {config['chunking']['type']} ({config['chunking']['params']})")
        st.write(f"• Retriever: {config['retriever']}")
        if config.get('hybrid_alpha'):
            st.write(f"• Hybrid alpha: {config['hybrid_alpha']}")
        st.write(f"• Documents: {result.get('document_count', 0)}")
        st.write(f"• Total Chunks: {result.get('total_chunks', 0)}")
    
    with col2:
        st.markdown("**Performance:**")
        st.write(f"• Total Time: {timing.get('total_time', 0):.2f}s")
        st.write(f"• Embedding: {timing.get('embedding_time', 0):.2f}s")
        
        # Enhanced retrieval info
        retrieval_candidates = result.get('retrieval_candidates', 0)
        retrieval_latency = timing.get('retrieval_latency_ms', 0)
        st.write(f"• Retrieval: {retrieval_candidates} candidates, {retrieval_latency:.1f}ms")
        
        st.write(f"• Generation: {timing.get('generation_time', 0):.2f}s")
        
        st.markdown("**Cost:**")
        st.write(f"• Embedding tokens: {result.get('embedding_tokens', 0)}")
        st.write(f"• Generation input tokens: {result.get('generation_input_tokens', 0)}")
        st.write(f"• Generation output tokens: {result.get('generation_output_tokens', 0)}")
    
    # Response
    st.markdown("**Generated Response:**")
    st.write(result.get('response', 'No response generated'))
    
    # Retrieved chunks (optional)
    retrieved_chunks = result.get('retrieved_chunks', [])
    if retrieved_chunks:
        with st.expander(f"📋 Retrieved Chunks (Top {len(retrieved_chunks)})", expanded=False):
            for i, chunk in enumerate(retrieved_chunks, 1):
                doc_id = chunk.get('document_id', chunk.get('doc', 'Unknown'))
                score = chunk.get('retrieval_score', chunk.get('score', 0))
                method = chunk.get('retrieval_method', 'unknown')
                
                st.write(f"**{i}. {doc_id}** (Score: {score:.3f}, Method: {method})")
                
                # Show chunk text preview
                chunk_text = chunk.get('text', '')
                if len(chunk_text) > 200:
                    chunk_text = chunk_text[:200] + "..."
                st.caption(chunk_text)
                
                # Show additional scores for hybrid results
                if method == 'hybrid':
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if 'vector_score' in chunk:
                            st.caption(f"Vector Score: {chunk['vector_score']:.3f}")
                    with col_b:
                        if 'bm25_score' in chunk:
                            st.caption(f"BM25 Score: {chunk['bm25_score']:.3f}")
                
                st.divider()

def main():
    """Main application"""
    logger.info("🚀 Starting RAG Indexing Comparison App")
    
    # Apply Amex GBT professional styling
    load_amex_styling()
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🔍 RAG Pipeline Comparison Platform")
    st.markdown("### Enterprise-grade analysis of document retrieval and AI response generation")
    st.markdown("*Compare different AI models, document processing strategies, and search methods to optimize your knowledge retrieval system*")
    st.divider()
    
    # Check if setup validation was run
    if not os.path.exists('.env'):
        st.error("⚠️ Environment not configured. Please run `python setup_validation.py` first.")
        logger.error("❌ .env file not found")
        st.stop()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    with st.container():
        # Document upload
        docs_uploaded = render_document_upload()
        
        # Query section
        query, run_comparison = render_query_section()
        
        # Results section (if any)
        if st.session_state.pipeline_results:
            render_results()
        
        # Footer
        st.markdown("---")
        st.markdown("💡 **Tip:** Results are automatically logged to Google Sheets for offline analysis")

if __name__ == "__main__":
    main()
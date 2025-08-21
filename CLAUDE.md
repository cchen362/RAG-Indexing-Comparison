# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup and Validation
```bash
# Setup virtual environment
python -m venv rag-env
rag-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Validate setup (test API keys and Google Sheets)
python setup_validation.py
```

### Running the Application
```bash
# Run main Streamlit app
streamlit run app.py

# Test individual components
python embedding_pipeline.py
python retrieval_system.py
python sheets_logger.py
```

### Environment Configuration
Required `.env` file in project root:
```env
OPENAI_API_KEY=sk-your-key-here
COHERE_API_KEY=your-key-here
GOOGLE_SHEET_ID=your-sheet-id
```

## Architecture Overview

This is a **modular RAG pipeline comparison system** that allows side-by-side testing of different retrieval-augmented generation configurations. The architecture follows a **pipeline orchestration pattern** where each component is independent and can be configured dynamically.

### Core Pipeline Flow
1. **Document Upload** → **Text Extraction** → **Chunking** → **Embedding** → **Indexing** → **Query Processing** → **Retrieval** → **Response Generation** → **Results Display** → **Logging**

### Key Architectural Components

#### Pipeline Orchestrator (`rag_pipeline.py`)
- **RAGPipeline**: Main orchestrator that coordinates all components
- Handles configuration-based execution for multiple pipeline variants
- Manages timing metrics and error handling across the entire flow
- Integrates with Streamlit session state for UI coordination

#### Modular Component System
- **DocumentProcessor**: Handles PDF/DOCX/TXT parsing with pluggable chunking strategies
- **EmbeddingPipeline**: Abstracts OpenAI and Cohere embedding APIs with auto-dimension detection
- **RetrievalSystem**: Implements vector (FAISS), BM25, and hybrid retrieval with dynamic indexing
- **SheetsLogger**: Background logging to Google Sheets with structured metadata

#### Configuration-Driven Design
The system uses **configuration dictionaries** that specify:
```python
config = {
    'embedding': {'provider': 'openai', 'model': 'text-embedding-3-small', 'dimension': 1536},
    'chunking': {'type': 'fixed', 'params': '500c-50o'},
    'retriever': 'hybrid'
}
```

Each configuration generates a unique pipeline execution path, enabling systematic comparison.

#### Dynamic Adaptation
- **Auto-dimension detection**: FAISS indices adapt to actual embedding dimensions returned by APIs
- **Graceful degradation**: Components handle API failures and dimension mismatches
- **Session persistence**: Streamlit maintains pipeline state across UI interactions

### Embedding Provider Integration
- **OpenAI**: Direct API integration with text-embedding-3-small/large models
- **Cohere**: Uses ClientV2 with embed-v4.0, auto-detects dimensions (typically 1536d default)
- **Provider abstraction**: EmbeddingProvider base class allows easy extension

### Retrieval System Architecture
- **Vector retrieval**: FAISS IndexFlatIP with cosine similarity and L2 normalization
- **BM25 retrieval**: Custom implementation with configurable k1/b parameters
- **Hybrid retrieval**: Weighted combination (70% vector + 30% BM25) with score normalization
- **Parallel indexing**: Each retrieval method builds independent indices

### Response Generation
- **Fixed model approach**: Uses GPT-4o-mini for all configurations to ensure fair comparison
- **Context injection**: Retrieved chunks are formatted with source attribution and relevance scores
- **Error resilience**: Handles API failures with informative error messages

### Google Sheets Integration
- **Structured logging**: Each pipeline run generates a comprehensive row with timing, configuration, and results
- **Service account authentication**: Uses JSON credentials file for automated access
- **Schema management**: Auto-creates headers and handles sheet formatting

## Important Implementation Details

### Virtual Environment Required
This project uses a virtual environment (`rag-env`) to isolate dependencies. Always activate before development:
```bash
rag-env\Scripts\activate
```

### API Key Management
- API keys are loaded from `.env` file using python-dotenv
- Keys are validated on startup via `setup_validation.py`
- No API keys should be hardcoded in source files

### Google Sheets Setup
- Requires service account JSON file in `credentials/` directory
- Target spreadsheet must be shared with service account email
- Sheet ID is configured in `.env` or defaults to hardcoded value

### Streamlit Session State
The UI heavily relies on Streamlit session state for:
- Pipeline instance persistence (`st.session_state.rag_pipeline`)
- Configuration selections across sidebar interactions
- Results storage and display (`st.session_state.pipeline_results`)

### Error Handling Pattern
Components use a consistent error handling approach:
- Catch exceptions at component boundaries
- Return structured error results with timing data
- Display user-friendly error messages while preserving technical details for debugging

### Performance Considerations
- Embedding APIs are called in batches (OpenAI: 100, Cohere: 96)
- FAISS operations use float32 arrays with contiguous memory layout
- UI updates use progress tracking for long-running operations
- Results display is optimized for large numbers of configurations

## Common Development Patterns

### Adding New Embedding Providers
1. Extend `EmbeddingProvider` base class
2. Add provider initialization in `EmbeddingPipeline._initialize_providers()`
3. Update `get_provider()` method for routing
4. Add UI checkbox in `app.py` sidebar

### Adding New Retrieval Methods
1. Create retriever class with `build_index()` and `search()` methods
2. Integrate into `RetrievalSystem.build_index()` and `search()`
3. Update configuration generation in `get_active_configurations()`

### Configuration Testing
Use the test functions in individual modules (`test_embedding_pipeline()`, `test_retrieval_system()`) to validate component functionality before integration.
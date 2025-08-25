# RAG Indexing Comparison App

A production-ready Streamlit application for testing and comparing different Retrieval-Augmented Generation (RAG) pipeline configurations side-by-side.

**🌐 Live Demo:** https://ragindex.zyroi.com  
**📊 Status:** Production deployment active since August 2025

## 🎯 Purpose

This app allows you to:
- Upload documents (PDF, DOCX, TXT)
- Select different embedding models, chunking strategies, and retrieval methods
- Run queries across multiple configurations simultaneously
- Compare performance and quality of different RAG setups
- Automatically log all results to Google Sheets for analysis

## 🚀 Quick Start

### Option 1: Use Live Demo (Recommended)
Visit **https://ragindex.zyroi.com** to use the application immediately without any setup.

### Option 2: Local Development Setup

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Set Up Environment Variables
Create a `.env` file in the project root:
```env
# OpenAI API Key
OPENAI_API_KEY=sk-your-openai-key-here

# Cohere API Key  
COHERE_API_KEY=your-cohere-key-here

# Google Sheets ID (for logging results)
GOOGLE_SHEET_ID=your-sheet-id-here
```

#### 3. Set Up Google Sheets (Optional)
For result logging, place your service account JSON in:
```
credentials/rag-comparison-app-*.json
```

#### 4. Validate Setup
```bash
python setup_validation.py
```

#### 5. Run Locally
```bash
streamlit run app.py
```

### Option 3: Production Deployment
For Docker deployment on your own server:
```bash
# Quick deploy to production server
chmod +x quick_deploy.sh
./quick_deploy.sh
```

See `DEPLOYMENT_GUIDE.md` for detailed deployment instructions.

## 📋 Features

### Embedding Models
- **OpenAI**: text-embedding-3-small (1536d), text-embedding-3-large (3072d)
- **Cohere**: embed-v4.0 (512d, 1024d)

### Chunking Strategies
- **Fixed-size**: Character-based chunking with configurable size and overlap
- **Sentence-based**: Preserve sentence boundaries with configurable sentence count

### Retrieval Methods
- **Vector-only**: Cosine similarity using FAISS
- **BM25-only**: Keyword-based retrieval  
- **Hybrid**: Weighted combination (70% vector + 30% BM25)

### Response Generation
- **GPT-4o-mini**: Fixed model for consistent comparison across retrieval methods

## 🔧 Configuration Options

### Sidebar Controls
- **Embedding Models**: Select multiple models to compare dimensions and providers
- **Chunking Strategy**: Choose strategy and configure parameters
- **Retrieval Method**: Select vector, keyword, or hybrid approaches
- **Top-k Results**: Number of chunks to retrieve (1-20)

### Default Configuration
The app starts with an optimal configuration:
- OpenAI text-embedding-3-small (1536d)
- Cohere embed-v4.0 (512d)  
- Fixed chunking (500 chars, 50 overlap)
- Vector-only and Hybrid retrieval

## 📊 Results & Analysis

### Real-time Display
- Configuration details and performance metrics
- Generated responses for each setup
- Retrieved chunks with scores and relevance
- Timing breakdown (chunking, embedding, indexing, retrieval, generation)

### Google Sheets Logging
All results are automatically logged with:
- Timestamp and unique run ID
- Query and configuration details
- Performance metrics and timing
- Response text and retrieved chunks
- Metadata for offline analysis

## 🗂️ File Structure

```
RAG-Indexing-Comparison/
├── app.py                          # Main Streamlit application
├── rag_pipeline.py                 # Core pipeline orchestrator  
├── document_processor.py           # Document parsing and chunking
├── embedding_pipeline.py           # Embedding model integrations
├── retrieval_system.py             # Vector, BM25, and hybrid retrieval
├── sheets_logger.py                # Google Sheets logging
├── setup_validation.py             # Setup and API validation
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── .gitignore                     # Git ignore file
├── credentials/                   # Google Sheets credentials
│   └── rag-comparison-app-*.json  
├── logs/                          # Application logs (auto-created)
├── Dockerfile                     # Docker container definition
├── docker-compose.yml             # Docker orchestration
├── deploy.sh                      # Local deployment script
├── quick_deploy.sh               # SSH deployment script
├── encode_credentials.py          # Credential encoding utility
├── validate_deployment_simple.py  # Deployment validation
├── README.md                      # This file
├── DEPLOYMENT.md                  # Deployment guide
└── DEPLOYMENT_GUIDE.md           # Comprehensive deployment docs
```

## 🔍 Usage Examples

### Basic Comparison
1. Upload a document (PDF, DOCX, or TXT)
2. Enter a query like "What is machine learning?"
3. Select 2-3 configurations from the sidebar
4. Click "Run Comparison"
5. Compare results in the output cards

### Advanced Testing
1. Upload multiple documents
2. Test different chunking strategies (fixed vs sentence-based)
3. Compare embedding dimensions (512d vs 1536d vs 3072d)
4. Evaluate retrieval methods (vector vs hybrid)
5. Analyze timing and quality trade-offs

## ⚙️ Technical Details

### Pipeline Architecture
1. **Document Processing**: Parse uploaded files and extract text
2. **Chunking**: Split documents using selected strategy
3. **Embedding**: Generate vectors using selected model
4. **Indexing**: Build FAISS vector index and BM25 index
5. **Query Processing**: Embed query and search indices
6. **Response Generation**: Use GPT-4o-mini with retrieved context
7. **Logging**: Save results to Google Sheets

### Performance Optimization
- Batch processing for embeddings
- Efficient FAISS indexing
- Parallel configuration execution
- Streaming results display

## 🛠️ Troubleshooting

### Common Issues

**API Key Errors**
- Verify keys in `.env` file
- Run `python setup_validation.py` to test

**Google Sheets Issues**  
- Check credentials file exists in `credentials/` folder
- Verify sheet is shared with service account email
- Test connection with setup validation

**Memory Issues**
- Reduce chunk size or document count
- Use smaller embedding dimensions
- Process fewer configurations simultaneously

**Slow Performance**
- Check internet connection for API calls
- Reduce top-k retrieval count
- Use smaller documents for testing

## 📈 Interpreting Results

### Performance Metrics
- **Total Time**: End-to-end pipeline execution
- **Embedding Time**: Vector generation duration
- **Retrieval Time**: Search and ranking duration
- **Generation Time**: LLM response creation

### Quality Indicators
- **Retrieval Scores**: Higher scores indicate better relevance
- **Response Accuracy**: Compare answers to expected results
- **Retrieved Chunks**: Review source material relevance

### Configuration Insights
- **Dimension Comparison**: Higher dimensions may improve accuracy but increase processing time
- **Chunking Strategy**: Sentence-based often provides better context than fixed-size
- **Retrieval Methods**: Hybrid typically outperforms vector-only for diverse queries

## 🚧 Future Enhancements

- Additional embedding providers (Anthropic, Hugging Face)
- Advanced chunking strategies (semantic, hierarchical)
- Reranker models integration
- Cost tracking and analysis
- Export functionality for results
- A/B testing framework

## 📄 License

This project is built for research and comparison purposes. Please ensure you comply with the terms of service for OpenAI, Cohere, and Google APIs when using this application.
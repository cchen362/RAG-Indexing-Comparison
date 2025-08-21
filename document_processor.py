"""
Document processing module for RAG Indexing Comparison App
Handles document parsing, text extraction, and chunking strategies
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pypdf
from docx import Document
import nltk
from nltk.tokenize import sent_tokenize
import re
import streamlit as st

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class DocumentProcessor:
    """Handles document parsing and text extraction"""
    
    def __init__(self):
        self.supported_extensions = {'.pdf', '.docx', '.txt'}
    
    def process_uploaded_files(self, uploaded_files) -> List[Dict[str, Any]]:
        """Process uploaded files and extract text content"""
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Get file extension
                file_extension = Path(uploaded_file.name).suffix.lower()
                
                if file_extension not in self.supported_extensions:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue
                
                # Extract text based on file type
                text_content = self._extract_text(uploaded_file, file_extension)
                
                if text_content:
                    documents.append({
                        'filename': uploaded_file.name,
                        'content': text_content,
                        'size': len(text_content),
                        'file_type': file_extension[1:]  # Remove the dot
                    })
                else:
                    st.warning(f"No text content extracted from: {uploaded_file.name}")
                    
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        return documents
    
    def _extract_text(self, uploaded_file, file_extension: str) -> str:
        """Extract text from different file types"""
        if file_extension == '.pdf':
            return self._extract_pdf_text(uploaded_file)
        elif file_extension == '.docx':
            return self._extract_docx_text(uploaded_file)
        elif file_extension == '.txt':
            return self._extract_txt_text(uploaded_file)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    
    def _extract_pdf_text(self, uploaded_file) -> str:
        """Extract text from PDF file using pypdf"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Read PDF
            text_content = []
            with open(tmp_file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(page_text)
                    except Exception as e:
                        st.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return '\n\n'.join(text_content)
            
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    def _extract_docx_text(self, uploaded_file) -> str:
        """Extract text from DOCX file"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Read DOCX
            doc = Document(tmp_file_path)
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return '\n\n'.join(text_content)
            
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {str(e)}")
    
    def _extract_txt_text(self, uploaded_file) -> str:
        """Extract text from TXT file"""
        try:
            # Decode the text content
            content = uploaded_file.getvalue()
            
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    return content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, use utf-8 with error handling
            return content.decode('utf-8', errors='replace')
            
        except Exception as e:
            raise Exception(f"TXT extraction failed: {str(e)}")


class ChunkingStrategy:
    """Base class for chunking strategies"""
    
    def chunk_text(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Chunk text into smaller pieces"""
        raise NotImplementedError


class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size character-based chunking with overlap"""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text into fixed-size chunks with overlap"""
        chunks = []
        text = text.strip()
        
        if not text:
            return chunks
        
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + chunk_size
            
            # If this is not the last chunk, try to break at word boundary
            if end < len(text):
                # Look for word boundary within the last 50 characters
                last_space = text.rfind(' ', start, end)
                if last_space > start + chunk_size - 50:
                    end = last_space
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end,
                    'size': len(chunk_text),
                    'strategy': 'fixed',
                    'params': f"{chunk_size}c-{overlap}o"
                })
                chunk_id += 1
            
            # Move start position (accounting for overlap)
            start = end - overlap if overlap > 0 else end
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks


class SentenceBasedChunking(ChunkingStrategy):
    """Sentence-based chunking that preserves sentence boundaries"""
    
    def chunk_text(self, text: str, max_sentences: int = 5) -> List[Dict[str, Any]]:
        """Split text into chunks based on sentences"""
        chunks = []
        text = text.strip()
        
        if not text:
            return chunks
        
        # Tokenize into sentences
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunk_id = 0
        current_chunk = []
        current_pos = 0
        
        for sentence in sentences:
            current_chunk.append(sentence)
            
            # If we've reached max sentences, create a chunk
            if len(current_chunk) >= max_sentences:
                chunk_text = ' '.join(current_chunk).strip()
                
                if chunk_text:
                    chunks.append({
                        'id': chunk_id,
                        'text': chunk_text,
                        'start_pos': current_pos,
                        'end_pos': current_pos + len(chunk_text),
                        'size': len(chunk_text),
                        'sentence_count': len(current_chunk),
                        'strategy': 'sentence',
                        'params': f"{max_sentences}s"
                    })
                    chunk_id += 1
                
                current_pos += len(chunk_text) + 1
                current_chunk = []
        
        # Handle remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'start_pos': current_pos,
                    'end_pos': current_pos + len(chunk_text),
                    'size': len(chunk_text),
                    'sentence_count': len(current_chunk),
                    'strategy': 'sentence',
                    'params': f"{max_sentences}s"
                })
        
        return chunks


class DocumentChunker:
    """Main document chunking orchestrator"""
    
    def __init__(self):
        self.strategies = {
            'fixed': FixedSizeChunking(),
            'sentence': SentenceBasedChunking()
        }
    
    def chunk_documents(self, documents: List[Dict[str, Any]], chunking_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk all documents using specified strategy"""
        all_chunks = []
        
        strategy_type = chunking_config.get('type', 'fixed')
        strategy = self.strategies.get(strategy_type)
        
        if not strategy:
            raise ValueError(f"Unknown chunking strategy: {strategy_type}")
        
        for doc in documents:
            # Extract chunking parameters
            if strategy_type == 'fixed':
                chunk_size = int(chunking_config.get('params', '500c-50o').split('c')[0])
                overlap = int(chunking_config.get('params', '500c-50o').split('-')[1].replace('o', ''))
                chunks = strategy.chunk_text(doc['content'], chunk_size=chunk_size, overlap=overlap)
            elif strategy_type == 'sentence':
                max_sentences = int(chunking_config.get('params', '5s').replace('s', ''))
                chunks = strategy.chunk_text(doc['content'], max_sentences=max_sentences)
            else:
                chunks = strategy.chunk_text(doc['content'])
            
            # Add document metadata to each chunk
            for chunk in chunks:
                chunk.update({
                    'document_id': doc['filename'],
                    'document_type': doc['file_type'],
                    'document_size': doc['size']
                })
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about the chunks"""
        if not chunks:
            return {}
        
        chunk_sizes = [chunk['size'] for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
            'strategies_used': list(set(chunk.get('strategy', 'unknown') for chunk in chunks))
        }


def preprocess_text(text: str) -> str:
    """Basic text preprocessing"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but preserve basic punctuation
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text
"""
Logging configuration for RAG Indexing Comparison App
Provides structured logging to both terminal and file
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import re

class SafeConsoleFormatter(logging.Formatter):
    """Formatter that removes emojis for console output on Windows"""
    
    def format(self, record):
        # Get the formatted message
        formatted = super().format(record)
        # Remove emojis for console (Windows compatibility)
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+')
        return emoji_pattern.sub('', formatted)

def setup_logging(log_level=logging.INFO, log_to_file=True):
    """Setup comprehensive logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = SafeConsoleFormatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler (terminal output) - safe for Windows
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if enabled) - keeps emojis
    if log_to_file:
        log_filename = f"rag_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_dir / log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # More detailed in file
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        print(f"Logging to: {log_dir / log_filename}")
    
    # Create specific loggers for different components
    loggers = {
        'rag_pipeline': logging.getLogger('rag_pipeline'),
        'embedding': logging.getLogger('embedding'),
        'retrieval': logging.getLogger('retrieval'),
        'document_processor': logging.getLogger('document_processor'),
        'sheets_logger': logging.getLogger('sheets_logger'),
        'streamlit_app': logging.getLogger('streamlit_app')
    }
    
    return loggers

# Global logger instance
def get_logger(name):
    """Get logger instance for specific component"""
    return logging.getLogger(name)
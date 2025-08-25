"""
Google Sheets logging module for RAG Indexing Comparison App
Handles automatic logging of pipeline results to Google Sheets
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import gspread
from google.oauth2.service_account import Credentials
import streamlit as st

class SheetsLogger:
    """Handles logging pipeline results to Google Sheets"""
    
    def __init__(self, credentials_path: str = None, sheet_id: str = None):
        self.credentials_path = credentials_path or 'credentials/rag-comparison-app-2ebc99e885e4.json'
        self.sheet_id = sheet_id or os.getenv('GOOGLE_SHEET_ID', '1U34uloZe1S0E-T83LDOtKfgYuBipBrejGdEW8QVSguI')
        self.gc = None
        self.sheet = None
        self.worksheet = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Google Sheets connection"""
        try:
            # Check if credentials file exists
            creds_path = Path(self.credentials_path)
            if not creds_path.exists():
                st.error(f"Google credentials file not found: {creds_path}")
                return False
            
            # Set up credentials and authorization
            scope = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            credentials = Credentials.from_service_account_file(
                str(creds_path), 
                scopes=scope
            )
            self.gc = gspread.authorize(credentials)
            
            # Open the spreadsheet
            self.sheet = self.gc.open_by_key(self.sheet_id)
            self.worksheet = self.sheet.sheet1
            
            # Initialize headers if needed
            self._ensure_headers()
            
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize Google Sheets connection: {str(e)}")
            return False
    
    def _ensure_headers(self):
        """Ensure the spreadsheet has proper headers"""
        expected_headers = [
            'Timestamp',
            'Run ID', 
            'Query',
            'Embedding Model',
            'Embedding Dimension',
            'Chunk Strategy',
            'Chunk Params',
            'Retriever',
            'Hybrid Alpha',
            'Total Time (s)',
            'Embedding Time (s)',
            'Embedding Tokens',
            'Retrieval Candidates',
            'Retrieval Latency (ms)',
            'Generation Time (s)',
            'Generation Input Tokens',
            'Generation Output Tokens',
            'Response',
            'Retrieved Chunks',
            'Config Name',
            'Document Count',
            'Total Chunks',
            'Top K',
            'Metadata'
        ]
        
        try:
            # Get current headers
            current_headers = self.worksheet.row_values(1)
            
            # If no headers or headers are different, update them
            if not current_headers or current_headers != expected_headers:
                self.worksheet.clear()
                self.worksheet.append_row(expected_headers)
                
                # Format header row
                self.worksheet.format('1:1', {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
                })
                
        except Exception as e:
            st.warning(f"Could not set up headers: {str(e)}")
    
    def log_pipeline_result(self, result: Dict[str, Any]) -> bool:
        """Log a single pipeline result to Google Sheets"""
        if not self.worksheet:
            st.warning("Google Sheets not initialized. Skipping logging.")
            return False
        
        try:
            # Extract data from result
            config = result.get('config', {})
            timing = result.get('timing', {})
            retrieved_chunks = result.get('retrieved_chunks', [])
            
            # Prepare row data
            row_data = [
                result.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                result.get('run_id', ''),
                result.get('query', ''),
                config.get('embedding', {}).get('model', ''),
                config.get('embedding', {}).get('dimension', ''),
                config.get('chunking', {}).get('type', ''),
                config.get('chunking', {}).get('params', ''),
                config.get('retriever', ''),
                config.get('hybrid_alpha', ''),
                timing.get('total_time', 0),
                timing.get('embedding_time', 0),
                result.get('embedding_tokens', 0),
                result.get('retrieval_candidates', 0),
                timing.get('retrieval_latency_ms', 0),
                timing.get('generation_time', 0),
                result.get('generation_input_tokens', 0),
                result.get('generation_output_tokens', 0),
                result.get('response', ''),
                self._format_retrieved_chunks(retrieved_chunks),
                config.get('name', ''),
                result.get('document_count', 0),
                result.get('total_chunks', 0),
                result.get('top_k', 5),
                json.dumps(result.get('metadata', {}), default=str)
            ]
            
            # Append row to sheet
            self.worksheet.append_row(row_data)
            
            return True
            
        except Exception as e:
            st.error(f"Failed to log result to Google Sheets: {str(e)}")
            return False
    
    def log_batch_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Log multiple pipeline results in batch"""
        successful_logs = 0
        failed_logs = 0
        
        for result in results:
            if self.log_pipeline_result(result):
                successful_logs += 1
            else:
                failed_logs += 1
        
        return {
            'total_results': len(results),
            'successful_logs': successful_logs,
            'failed_logs': failed_logs,
            'success_rate': successful_logs / len(results) if results else 0
        }
    
    def _format_retrieved_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks for display in sheets"""
        if not chunks:
            return ""
        
        formatted = []
        for i, chunk in enumerate(chunks[:3]):  # Limit to top 3 for readability
            doc_id = chunk.get('document_id', 'Unknown')
            score = chunk.get('retrieval_score', 0)
            text_preview = chunk.get('text', '')[:100] + '...' if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
            formatted.append(f"{i+1}. {doc_id} (Score: {score:.3f}): {text_preview}")
        
        return ' | '.join(formatted)
    
    def get_logged_runs_count(self) -> int:
        """Get the number of runs already logged"""
        if not self.worksheet:
            return 0
        
        try:
            # Get all values and count non-empty rows (excluding header)
            all_values = self.worksheet.get_all_values()
            return len(all_values) - 1 if all_values else 0
        except Exception:
            return 0
    
    def get_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent logged runs"""
        if not self.worksheet:
            return []
        
        try:
            # Get all records
            records = self.worksheet.get_all_records()
            
            # Return last N records
            recent_records = records[-limit:] if len(records) > limit else records
            
            # Reverse to show most recent first
            return list(reversed(recent_records))
            
        except Exception as e:
            st.warning(f"Could not fetch recent runs: {str(e)}")
            return []
    
    def clear_all_data(self) -> bool:
        """Clear all data from the sheet (keep headers)"""
        if not self.worksheet:
            return False
        
        try:
            # Get current headers
            headers = self.worksheet.row_values(1)
            
            # Clear all data
            self.worksheet.clear()
            
            # Restore headers
            if headers:
                self.worksheet.append_row(headers)
                self.worksheet.format('1:1', {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
                })
            
            return True
            
        except Exception as e:
            st.error(f"Failed to clear sheet data: {str(e)}")
            return False
    
    def create_summary_sheet(self, results: List[Dict[str, Any]]) -> bool:
        """Create a summary sheet with aggregated results"""
        if not self.sheet:
            return False
        
        try:
            # Check if summary sheet exists
            summary_sheet_name = f"Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            try:
                summary_sheet = self.sheet.worksheet(summary_sheet_name)
            except gspread.WorksheetNotFound:
                summary_sheet = self.sheet.add_worksheet(
                    title=summary_sheet_name,
                    rows=100,
                    cols=20
                )
            
            # Prepare summary data
            summary_data = self._calculate_summary_stats(results)
            
            # Add summary headers and data
            headers = [
                'Metric', 'Value', 'Details'
            ]
            summary_sheet.append_row(headers)
            
            # Add summary rows
            for metric, value, details in summary_data:
                summary_sheet.append_row([metric, value, details])
            
            # Format header
            summary_sheet.format('1:1', {
                'textFormat': {'bold': True},
                'backgroundColor': {'red': 0.8, 'green': 0.9, 'blue': 1.0}
            })
            
            return True
            
        except Exception as e:
            st.warning(f"Could not create summary sheet: {str(e)}")
            return False
    
    def _calculate_summary_stats(self, results: List[Dict[str, Any]]) -> List[tuple]:
        """Calculate summary statistics from results"""
        if not results:
            return []
        
        # Basic statistics
        total_runs = len(results)
        avg_total_time = sum(r.get('timing', {}).get('total_time', 0) for r in results) / total_runs
        
        # Group by configuration
        config_performance = {}
        for result in results:
            config_name = result.get('config', {}).get('name', 'Unknown')
            if config_name not in config_performance:
                config_performance[config_name] = []
            config_performance[config_name].append(result.get('timing', {}).get('total_time', 0))
        
        # Find best performing configuration
        best_config = min(config_performance.items(), key=lambda x: sum(x[1])/len(x[1]))[0]
        best_avg_time = sum(config_performance[best_config]) / len(config_performance[best_config])
        
        summary_stats = [
            ('Total Runs', total_runs, f'Across {len(config_performance)} configurations'),
            ('Average Total Time', f'{avg_total_time:.2f}s', 'All configurations'),
            ('Best Configuration', best_config, f'Avg time: {best_avg_time:.2f}s'),
            ('Total Configurations', len(config_performance), 'Unique config combinations'),
            ('Run Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Summary generated')
        ]
        
        return summary_stats
    
    def test_connection(self) -> bool:
        """Test the Google Sheets connection"""
        try:
            if not self.worksheet:
                return False
            
            # Try to read the first cell
            test_value = self.worksheet.cell(1, 1).value
            return True
            
        except Exception as e:
            st.error(f"Google Sheets connection test failed: {str(e)}")
            return False


def test_sheets_logger():
    """Test function for sheets logger"""
    logger = SheetsLogger()
    
    if not logger.test_connection():
        print("Sheets connection test failed")
        return False
    
    # Test data
    test_result = {
        'timestamp': datetime.now(),
        'run_id': 'test_001',
        'query': 'What is machine learning?',
        'config': {
            'name': 'openai-1536d_fixed-500c-50o_vector',
            'embedding': {
                'provider': 'openai',
                'model': 'text-embedding-3-small',
                'dimension': 1536
            },
            'chunking': {
                'type': 'fixed',
                'params': '500c-50o'
            },
            'retriever': 'vector'
        },
        'timing': {
            'total_time': 2.5,
            'embedding_time': 0.8,
            'retrieval_latency_ms': 15,
            'generation_time': 1.3
        },
        'embedding_tokens': 8,
        'retrieval_candidates': 50,
        'generation_input_tokens': 2847,
        'generation_output_tokens': 312,
        'response': 'Machine learning is a test response.',
        'retrieved_chunks': [
            {'document_id': 'test.pdf', 'retrieval_score': 0.85, 'text': 'Test chunk content'},
            {'document_id': 'test.pdf', 'retrieval_score': 0.72, 'text': 'Another test chunk'}
        ],
        'document_count': 1,
        'total_chunks': 10,
        'top_k': 5
    }
    
    try:
        success = logger.log_pipeline_result(test_result)
        if success:
            print("Test result logged successfully")
            return True
        else:
            print("Failed to log test result")
            return False
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_sheets_logger()
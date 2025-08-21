"""
Test script to verify logging works without Unicode errors
"""

from logger_config import setup_logging, get_logger

def test_logging():
    """Test logging with emojis"""
    # Setup logging
    setup_logging()
    logger = get_logger('test')
    
    # Test various emoji messages
    logger.info("🚀 Starting test...")
    logger.info("📄 Processing documents...")
    logger.info("🔧 Running configuration...")
    logger.info("✅ Test completed successfully!")
    
    print("If you see this message, logging is working correctly!")

if __name__ == "__main__":
    test_logging()
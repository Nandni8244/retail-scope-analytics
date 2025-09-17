import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseConfig:
    """Database configuration settings"""
    
    # PostgreSQL Configuration
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'retailscope_analytics')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
    
    @property
    def postgres_url(self):
        from urllib.parse import quote_plus
        # URL encode the password to handle special characters
        safe_password = quote_plus(self.POSTGRES_PASSWORD)
        return f"postgresql://{self.POSTGRES_USER}:{safe_password}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # MongoDB Configuration (for logs and real-time data)
    MONGO_HOST = os.getenv('MONGO_HOST', 'localhost')
    MONGO_PORT = os.getenv('MONGO_PORT', '27017')
    MONGO_DB = os.getenv('MONGO_DB', 'retailscope_logs')
    
    @property
    def mongo_url(self):
        return f"mongodb://{self.MONGO_HOST}:{self.MONGO_PORT}/{self.MONGO_DB}"

class APIConfig:
    """External API configuration"""
    
    # Sample e-commerce API (you can replace with real APIs)
    ECOMMERCE_API_KEY = os.getenv('ECOMMERCE_API_KEY', 'your_api_key_here')
    ECOMMERCE_BASE_URL = os.getenv('ECOMMERCE_BASE_URL', 'https://api.example-ecommerce.com/v1')
    
    # Rate limiting
    API_RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', '100'))  # requests per minute
    
class AppConfig:
    """Application configuration"""
    
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # File paths
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')
    
    # Ensure directories exist
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR]:
        os.makedirs(directory, exist_ok=True)

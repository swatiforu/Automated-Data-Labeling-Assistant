import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data_labeling.db")
    
    # App settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Default categories
    DEFAULT_CATEGORIES = [
        "positive", "negative", "neutral", "question", "statement"
    ]
    
    # Review settings
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.8"))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100")) 
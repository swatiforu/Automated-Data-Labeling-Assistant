import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data_labeling.db")
    
    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")  # local, huggingface, ollama, free_api
    
    # Local Model Settings
    LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "distilbert-base-uncased")
    USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"
    
    # Hugging Face Settings
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Optional, for private models
    HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "distilbert-base-uncased")
    
    # Ollama Settings (if using local Ollama)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
    
    # Free API Settings
    FREE_API_URL = os.getenv("FREE_API_URL", "https://api-inference.huggingface.co/models/")
    FREE_API_MODEL = os.getenv("FREE_API_MODEL", "distilbert-base-uncased")
    
    # App settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Default categories
    DEFAULT_CATEGORIES = [
        "positive", "negative", "neutral", "question", "statement",
        "complaint", "feedback", "inquiry", "other"
    ]
    
    # Review settings
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.8"))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))
    
    # Model confidence thresholds
    CONFIDENCE_THRESHOLDS = {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4
    } 
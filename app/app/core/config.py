"""
Configuration settings for the Employee Search RAG application.
"""

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_HOST: str = "127.0.0.1"  # or "localhost"

    API_PORT: int = 8000
    API_PREFIX: str = ""
    
    # Vector Store Settings
    VECTOR_STORE_PATH: str = "employee_faiss_index"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    SIMILARITY_THRESHOLD: float = 0.3
    MAX_RESULTS: int = 5
    
    # LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "mistral:7b"
    LLM_TEMPERATURE: float = 0.2
    LLM_CONTEXT_SIZE: int = 4096
    
    # Data Settings
    DATA_PATH: str = "data/employees.json"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 
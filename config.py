# -*- coding: utf-8 -*-
"""
Configuration settings for RAG Chatbot
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""
    
    # API Settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = "gpt-4.1-mini"
    
    # LLM Settings
    LLM_TEMPERATURE: float = 0.0
    LLM_TIMEOUT: int = 60
    
    # Document Processing Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval Settings
    RETRIEVAL_K: int = 3  # Number of documents to retrieve
    RETRIEVAL_SEARCH_TYPE: str = "similarity"
    
    # Embedding Settings
    EMBEDDING_DIMENSION: int = 384
    
    # File Upload Settings
    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_FILE_TYPES: list = ["pdf", "txt", "docx", "doc"]
    
    # UI Settings
    PAGE_TITLE: str = "RAG Chatbot"
    PAGE_ICON: str = "🤖"
    LAYOUT: str = "wide"
    
    # Text Display Settings
    SOURCE_PREVIEW_LENGTH: int = 300
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.OPENAI_API_KEY:
            print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
            return False
        return True


# Development configuration
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"


# Production configuration
class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"


# Get configuration based on environment
ENV = os.getenv("ENVIRONMENT", "development").lower()
if ENV == "production":
    config = ProductionConfig()
else:
    config = DevelopmentConfig()

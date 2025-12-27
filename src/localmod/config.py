"""Configuration management for LocalMod."""

from functools import lru_cache
from typing import List
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Model settings
    device: str = "auto"
    model_cache_dir: str = "~/.cache/localmod"
    lazy_load: bool = True
    
    # Classifier defaults
    toxicity_threshold: float = 0.5
    pii_threshold: float = 0.5
    prompt_injection_threshold: float = 0.5
    spam_threshold: float = 0.5
    nsfw_threshold: float = 0.5
    
    # API settings
    enable_docs: bool = True
    cors_origins: List[str] = ["*"]
    rate_limit: int = 100  # requests per minute
    max_batch_size: int = 32
    max_text_length: int = 10000
    
    # Logging
    log_level: str = "INFO"
    log_requests: bool = True

    class Config:
        env_prefix = "LOCALMOD_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

"""Configuration management for LocalMod."""

from functools import lru_cache
import sys

# Support both pydantic v1 and v2
# Pydantic v2 moved BaseSettings to pydantic-settings package
try:
    from pydantic_settings import BaseSettings
    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import BaseSettings  # type: ignore[no-redef]
        PYDANTIC_V2 = False
    except ImportError:
        raise ImportError(
            "Could not import BaseSettings. "
            "For pydantic v2, install pydantic-settings: pip install pydantic-settings"
        )

# Use List from typing for Python 3.6/3.7 compatibility
if sys.version_info >= (3, 9):
    from typing import List
else:
    from typing import List


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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

"""Configuration management for LocalMod."""

import os
from functools import lru_cache
import sys

# Support both pydantic v1 and v2
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
    model_dir: str = ""  # If empty, uses <model_cache_dir>/models
    lazy_load: bool = True
    offline: bool = False  # Strict offline mode - fail if models not local
    
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

def get_model_dir() -> str:
    """
    Get the model directory path.
    
    Priority:
    1. LOCALMOD_MODEL_DIR environment variable
    2. Settings.model_dir if set
    3. <model_cache_dir>/models (default: ~/.cache/localmod/models)
    """
    # Check environment variable first
    env_model_dir = os.environ.get("LOCALMOD_MODEL_DIR")
    if env_model_dir:
        return os.path.expanduser(env_model_dir)
    
    settings = get_settings()
    
    # Use explicit model_dir if set
    if settings.model_dir:
        return os.path.expanduser(settings.model_dir)
    
    # Default to <model_cache_dir>/models
    return os.path.join(os.path.expanduser(settings.model_cache_dir), "models")


def is_offline_mode() -> bool:
    """
    Check if offline mode is enabled.
    
    Respects:
    - LOCALMOD_OFFLINE=1 or true
    - HF_HUB_OFFLINE=1
    - TRANSFORMERS_OFFLINE=1
    """
    # Check LOCALMOD_OFFLINE
    offline_env = os.environ.get("LOCALMOD_OFFLINE", "").lower()
    if offline_env in ("1", "true", "yes"):
        return True
    
    # Check HuggingFace offline env vars
    if os.environ.get("HF_HUB_OFFLINE", "") == "1":
        return True
    if os.environ.get("TRANSFORMERS_OFFLINE", "") == "1":
        return True
    
    # Check settings
    settings = get_settings()
    return settings.offline

    return Settings()


def get_model_dir() -> str:
    """
    Get the model directory path.
    
    Priority:
    1. LOCALMOD_MODEL_DIR environment variable
    2. Settings.model_dir if set
    3. <model_cache_dir>/models (default: ~/.cache/localmod/models)
    """
    # Check environment variable first
    env_model_dir = os.environ.get("LOCALMOD_MODEL_DIR")
    if env_model_dir:
        return os.path.expanduser(env_model_dir)
    
    settings = get_settings()
    
    # Use explicit model_dir if set
    if settings.model_dir:
        return os.path.expanduser(settings.model_dir)
    
    # Default to <model_cache_dir>/models
    return os.path.join(os.path.expanduser(settings.model_cache_dir), "models")


def is_offline_mode() -> bool:
    """
    Check if offline mode is enabled.
    
    Respects:
    - LOCALMOD_OFFLINE=1 or true
    - HF_HUB_OFFLINE=1
    - TRANSFORMERS_OFFLINE=1
    """
    # Check LOCALMOD_OFFLINE
    offline_env = os.environ.get("LOCALMOD_OFFLINE", "").lower()
    if offline_env in ("1", "true", "yes"):
        return True
    
    # Check HuggingFace offline env vars
    if os.environ.get("HF_HUB_OFFLINE", "") == "1":
        return True
    if os.environ.get("TRANSFORMERS_OFFLINE", "") == "1":
        return True
    
    # Check settings
    settings = get_settings()
    return settings.offline


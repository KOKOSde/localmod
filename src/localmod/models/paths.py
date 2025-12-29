"""Model path utilities for LocalMod."""

import os
from typing import Dict, Optional

from localmod.config import get_model_dir, is_offline_mode


# Canonical model registry - maps classifier names to HuggingFace model IDs
MODEL_REGISTRY: Dict[str, str] = {
    "toxicity": "unitary/toxic-bert",
    "prompt_injection": "deepset/deberta-v3-base-injection",
    "spam": "mshenoda/roberta-spam",
    "nsfw": "michellejieli/NSFW_text_classifier",
    # PII uses regex, no ML model needed
}

# Order for downloading (stable, user-facing)
DOWNLOAD_ORDER = ["toxicity", "prompt_injection", "spam", "nsfw"]


def get_classifier_model_path(classifier_name: str) -> str:
    """
    Get the path to load a classifier's model.
    
    In offline mode: Returns local path only, raises error if not found.
    In online mode: Returns local path if exists, otherwise HuggingFace model ID.
    
    Args:
        classifier_name: Name of the classifier (toxicity, spam, etc.)
        
    Returns:
        Path to local model directory or HuggingFace model ID
        
    Raises:
        FileNotFoundError: If offline mode and model not found locally
    """
    if classifier_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown classifier: {classifier_name}. "
                        f"Available: {list(MODEL_REGISTRY.keys())}")
    
    model_dir = get_model_dir()
    local_path = os.path.join(model_dir, classifier_name)
    
    # Check if local model exists
    local_exists = (
        os.path.isdir(local_path) and 
        os.path.exists(os.path.join(local_path, "config.json"))
    )
    
    if is_offline_mode():
        if not local_exists:
            raise FileNotFoundError(
                f"Model for '{classifier_name}' not found at {local_path}. "
                f"Run 'localmod download' first, or set LOCALMOD_OFFLINE=0."
            )
        return local_path
    
    # Online mode: prefer local if exists, else use HuggingFace
    if local_exists:
        return local_path
    
    return MODEL_REGISTRY[classifier_name]


def get_local_model_path(classifier_name: str) -> str:
    """
    Get the local directory path for a classifier's model.
    
    Does not check if the model exists - use for downloading.
    
    Args:
        classifier_name: Name of the classifier
        
    Returns:
        Path to local model directory
    """
    model_dir = get_model_dir()
    return os.path.join(model_dir, classifier_name)


def model_exists_locally(classifier_name: str) -> bool:
    """Check if a model exists in the local model directory."""
    local_path = get_local_model_path(classifier_name)
    return (
        os.path.isdir(local_path) and 
        os.path.exists(os.path.join(local_path, "config.json"))
    )


def get_hf_model_id(classifier_name: str) -> str:
    """Get the HuggingFace model ID for a classifier."""
    if classifier_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    return MODEL_REGISTRY[classifier_name]


def get_transformers_kwargs() -> Dict[str, bool]:
    """
    Get kwargs for transformers from_pretrained() based on offline mode.
    
    Returns:
        Dict with local_files_only=True if offline mode enabled
    """
    if is_offline_mode():
        return {"local_files_only": True}
    return {}


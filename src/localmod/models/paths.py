"""Model path resolution utilities for LocalMod.

Handles:
- Finding local models in LOCALMOD_MODEL_DIR or ~/.cache/localmod/models
- Falling back to HuggingFace model IDs if local not found
- Offline mode via LOCALMOD_OFFLINE environment variable
"""

import os
from typing import Dict, Any

# Model registry: classifier name -> HuggingFace model ID
MODEL_REGISTRY: Dict[str, str] = {
    # Toxicity ensemble models (weighted: 50% unitary, 20% dehatebert, 15% snlp, 15% facebook)
    "toxicity": "unitary/toxic-bert",
    "toxicity_dehatebert": "Hate-speech-CNERG/dehatebert-mono-english",
    "toxicity_snlp": "s-nlp/roberta_toxicity_classifier",
    "toxicity_facebook": "facebook/roberta-hate-speech-dynabench-r4-target",
    # Other classifiers
    "spam": "mshenoda/roberta-spam",
    "prompt_injection": "deepset/deberta-v3-base-injection",
    "nsfw": "michellejieli/NSFW_text_classifier",
}

# Toxicity ensemble model names
TOXICITY_ENSEMBLE_MODELS = ["toxicity", "toxicity_dehatebert", "toxicity_snlp", "toxicity_facebook"]

# Toxicity ensemble weights (sum to 1.0)
TOXICITY_ENSEMBLE_WEIGHTS = {
    "toxicity": 0.50,           # Unitary Toxic-BERT
    "toxicity_dehatebert": 0.20, # DeHateBERT
    "toxicity_snlp": 0.15,       # s-nlp RoBERTa
    "toxicity_facebook": 0.15,   # Facebook Dynabench
}

# Stable download order
DOWNLOAD_ORDER = [
    "toxicity", "toxicity_dehatebert", "toxicity_snlp", "toxicity_facebook",
    "prompt_injection", "spam", "nsfw"
]


def get_model_dir() -> str:
    """Get the base directory for model storage."""
    model_dir = os.environ.get("LOCALMOD_MODEL_DIR")
    if model_dir:
        return os.path.expanduser(model_dir)
    
    # Default: ~/.cache/localmod/models
    cache_dir = os.environ.get("LOCALMOD_CACHE_DIR", "~/.cache/localmod")
    return os.path.expanduser(os.path.join(cache_dir, "models"))


def get_local_model_path(classifier_name: str) -> str:
    """Get the local model path for a classifier."""
    return os.path.join(get_model_dir(), classifier_name)


def model_exists_locally(classifier_name: str) -> bool:
    """Check if a model exists locally."""
    local_path = get_local_model_path(classifier_name)
    # Check for common model files
    return (
        os.path.isdir(local_path) and
        (
            os.path.exists(os.path.join(local_path, "config.json")) or
            os.path.exists(os.path.join(local_path, "pytorch_model.bin")) or
            os.path.exists(os.path.join(local_path, "model.safetensors"))
        )
    )


def get_hf_model_id(classifier_name: str) -> str:
    """Get the HuggingFace model ID for a classifier."""
    if classifier_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    return MODEL_REGISTRY[classifier_name]


def is_offline_mode() -> bool:
    """Check if offline mode is enabled."""
    offline = os.environ.get("LOCALMOD_OFFLINE", "").lower()
    if offline in ("1", "true", "yes"):
        return True
    # Also check HuggingFace offline vars
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return True
    if os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        return True
    return False


def get_classifier_model_path(classifier_name: str) -> str:
    """
    Get the model path to use for a classifier.
    
    Priority:
    1. Local path if exists and offline mode is enabled
    2. Local path if exists
    3. HuggingFace model ID (if not offline)
    4. Raise error if offline and no local model
    """
    local_path = get_local_model_path(classifier_name)
    hf_model_id = get_hf_model_id(classifier_name)
    offline = is_offline_mode()
    
    if model_exists_locally(classifier_name):
        return local_path
    
    if offline:
        raise FileNotFoundError(
            f"Offline mode enabled but model '{classifier_name}' not found at '{local_path}'. "
            f"Run 'localmod download' to download models first."
        )
    
    # Return HuggingFace model ID for download
    return hf_model_id


def get_transformers_kwargs() -> Dict[str, Any]:
    """Get kwargs to pass to transformers from_pretrained methods."""
    kwargs: Dict[str, Any] = {}
    
    if is_offline_mode():
        kwargs["local_files_only"] = True
    
    return kwargs





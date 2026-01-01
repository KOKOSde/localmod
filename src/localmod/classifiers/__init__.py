"""Classifier registry and exports."""

from typing import Dict, Type

from localmod.classifiers.toxicity import ToxicityClassifier
from localmod.classifiers.pii import PIIDetector
from localmod.classifiers.prompt_injection import PromptInjectionDetector
from localmod.classifiers.spam import SpamClassifier
from localmod.classifiers.nsfw import NSFWClassifier
from localmod.classifiers.nsfw_image import ImageNSFWClassifier
from localmod.models.base import BaseClassifier, ClassificationResult, Severity

__all__ = [
    "ToxicityClassifier",
    "PIIDetector", 
    "PromptInjectionDetector",
    "SpamClassifier",
    "NSFWClassifier",
    "ImageNSFWClassifier",
    "BaseClassifier",
    "ClassificationResult",
    "Severity",
    "CLASSIFIER_REGISTRY",
    "IMAGE_CLASSIFIER_REGISTRY",
    "get_classifier",
]

# Text classifiers
CLASSIFIER_REGISTRY: Dict[str, Type[BaseClassifier]] = {
    "toxicity": ToxicityClassifier,
    "pii": PIIDetector,
    "prompt_injection": PromptInjectionDetector,
    "spam": SpamClassifier,
    "nsfw": NSFWClassifier,
}

# Image classifiers
IMAGE_CLASSIFIER_REGISTRY: Dict[str, Type[BaseClassifier]] = {
    "nsfw_image": ImageNSFWClassifier,
}


def get_classifier(name: str, **kwargs) -> BaseClassifier:
    """Get classifier instance by name."""
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(f"Unknown classifier: {name}. Available: {list(CLASSIFIER_REGISTRY.keys())}")
    return CLASSIFIER_REGISTRY[name](**kwargs)

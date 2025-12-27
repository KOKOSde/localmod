"""LocalMod - Fully offline content moderation API."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your@email.com"

from localmod.pipeline import SafetyPipeline, SafetyReport
from localmod.classifiers import (
    BaseClassifier,
    ClassificationResult,
    Severity,
    ToxicityClassifier,
    PIIDetector,
    PromptInjectionDetector,
    SpamClassifier,
    NSFWClassifier,
)

__all__ = [
    "__version__",
    "SafetyPipeline",
    "SafetyReport",
    "BaseClassifier",
    "ClassificationResult",
    "Severity",
    "ToxicityClassifier",
    "PIIDetector",
    "PromptInjectionDetector",
    "SpamClassifier",
    "NSFWClassifier",
]

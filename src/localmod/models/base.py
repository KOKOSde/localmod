"""Base classifier interface for all LocalMod classifiers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    """Severity levels for flagged content."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ClassificationResult:
    """Result from a single classifier."""
    classifier: str
    flagged: bool
    confidence: float
    severity: Severity
    categories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classifier": self.classifier,
            "flagged": self.flagged,
            "confidence": round(self.confidence, 4),
            "severity": self.severity.value,
            "categories": self.categories,
            "metadata": self.metadata,
            "explanation": self.explanation,
        }


class BaseClassifier(ABC):
    """Abstract base class for all content classifiers."""

    name: str = "base"
    version: str = "0.0.0"
    
    def __init__(self, device: str = "auto", threshold: float = 0.5):
        self.threshold = threshold
        self._device = self._resolve_device(device)
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Called lazily on first prediction."""
        pass

    @abstractmethod
    def predict(self, text: str) -> ClassificationResult:
        """Classify a single text input."""
        pass

    def predict_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts. Override for optimized batching."""
        return [self.predict(text) for text in texts]

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before prediction."""
        if not self.is_loaded:
            self.load()

    def get_explanation(self, text: str, result: ClassificationResult) -> str:
        """Generate human-readable explanation for the classification."""
        if not result.flagged:
            return f"Content passed {self.name} check (confidence: {1 - result.confidence:.2%})"
        return (
            f"Content flagged by {self.name} "
            f"(confidence: {result.confidence:.2%}, severity: {result.severity.value})"
        )

"""NSFW (Not Safe For Work) text content detection."""

from typing import List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from localmod.models.base import BaseClassifier, ClassificationResult, Severity


class NSFWClassifier(BaseClassifier):
    """
    Detects sexually explicit or adult content in text.
    
    Uses a fine-tuned model for NSFW text classification.
    """
    
    name = "nsfw"
    version = "1.0.0"
    
    MODEL_NAME = "michellejieli/NSFW_text_classifier"
    
    NSFW_CATEGORIES = [
        "sexual_content",
        "adult_themes",
        "explicit_language",
    ]

    def __init__(
        self,
        device: str = "auto",
        threshold: float = 0.5,
    ):
        super().__init__(device=device, threshold=threshold)

    def load(self) -> None:
        """Load NSFW detection model."""
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        )
        self._model.to(self._device)
        self._model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> ClassificationResult:
        """Detect NSFW content in text."""
        self._ensure_loaded()
        
        if not text.strip():
            return ClassificationResult(
                classifier=self.name,
                flagged=False,
                confidence=0.0,
                severity=Severity.NONE,
            )
        
        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self._device)
        
        # Inference
        outputs = self._model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        
        # Get NSFW probability
        # Model typically outputs [SFW, NSFW] or similar
        if probs.shape[-1] == 2:
            nsfw_prob = probs[0, 1].item()
        else:
            nsfw_prob = probs[0].max().item()
        
        flagged = nsfw_prob >= self.threshold
        
        return ClassificationResult(
            classifier=self.name,
            flagged=flagged,
            confidence=nsfw_prob,
            severity=self._get_severity(nsfw_prob),
            categories=["sexual_content"] if flagged else [],
            metadata={"model": self.MODEL_NAME},
        )

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Batch prediction for efficiency."""
        self._ensure_loaded()
        
        if not texts:
            return []
        
        # Filter and track valid texts
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            return [
                ClassificationResult(
                    classifier=self.name,
                    flagged=False,
                    confidence=0.0,
                    severity=Severity.NONE,
                )
                for _ in texts
            ]
        
        # Batch tokenize
        inputs = self._tokenizer(
            valid_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self._device)
        
        # Batch inference
        outputs = self._model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        
        # Build results
        results: List[Optional[ClassificationResult]] = [None] * len(texts)
        
        for idx, orig_idx in enumerate(valid_indices):
            if probs.shape[-1] == 2:
                nsfw_prob = probs[idx, 1].item()
            else:
                nsfw_prob = probs[idx].max().item()
            
            flagged = nsfw_prob >= self.threshold
            results[orig_idx] = ClassificationResult(
                classifier=self.name,
                flagged=flagged,
                confidence=nsfw_prob,
                severity=self._get_severity(nsfw_prob),
                categories=["sexual_content"] if flagged else [],
                metadata={"model": self.MODEL_NAME},
            )
        
        # Fill empty results
        for i, result in enumerate(results):
            if result is None:
                results[i] = ClassificationResult(
                    classifier=self.name,
                    flagged=False,
                    confidence=0.0,
                    severity=Severity.NONE,
                )
        
        return results  # type: ignore

    def _get_severity(self, confidence: float) -> Severity:
        """Map confidence to severity."""
        if confidence < self.threshold:
            return Severity.NONE
        elif confidence < 0.6:
            return Severity.LOW
        elif confidence < 0.75:
            return Severity.MEDIUM
        elif confidence < 0.9:
            return Severity.HIGH
        else:
            return Severity.CRITICAL


"""Spam and promotional content detection."""

import re
from typing import Dict, List, Optional, Pattern, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from localmod.models.base import BaseClassifier, ClassificationResult, Severity
from localmod.models.paths import get_classifier_model_path, get_transformers_kwargs


class SpamClassifier(BaseClassifier):
    """
    Detects spam, promotional content, and scam messages.
    
    Uses a combination of heuristics and ML model for detection.
    """
    
    name = "spam"
    version = "1.0.0"
    
    # Spam indicators (heuristics)
    SPAM_PATTERNS: Dict[str, str] = {
        "excessive_caps": r'[A-Z]{5,}',
        "excessive_punctuation": r'[!?]{3,}',
        "money_symbols": r'[$€£¥]\s*\d+|\d+\s*[$€£¥]',
        "urgency_words": r'\b(urgent|immediately|act\s+now|limited\s+time|expires?)\b',
        "free_offers": r'\b(free|winner|won|congratulations|claim\s+your)\b',
        "click_bait": r'\b(click\s+here|sign\s+up|subscribe|buy\s+now)\b',
        "suspicious_urls": r'(bit\.ly|tinyurl|goo\.gl|t\.co|shorturl)',
    }
    
    # Weight for each pattern
    PATTERN_WEIGHTS: Dict[str, float] = {
        "excessive_caps": 0.1,
        "excessive_punctuation": 0.1,
        "money_symbols": 0.15,
        "urgency_words": 0.2,
        "free_offers": 0.2,
        "click_bait": 0.15,
        "suspicious_urls": 0.25,
    }

    def __init__(
        self,
        device: str = "auto",
        threshold: float = 0.5,
        use_ml_model: bool = True,
    ):
        super().__init__(device=device, threshold=threshold)
        self.use_ml_model = use_ml_model
        self._compiled_patterns: Dict[str, Pattern] = {}
        self._model_path: Optional[str] = None

    def load(self) -> None:
        """Load spam detection model and compile patterns."""
        # Compile patterns
        for name, pattern in self.SPAM_PATTERNS.items():
            self._compiled_patterns[name] = re.compile(pattern, re.IGNORECASE)
        
        # Load ML model if enabled
        if self.use_ml_model:
            try:
                self._model_path = get_classifier_model_path("spam")
                kwargs = get_transformers_kwargs()
                
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, **kwargs)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_path, **kwargs
                )
                self._model.to(self._device)
                self._model.eval()
            except Exception:
                self.use_ml_model = False
                self._model = True
        else:
            self._model = True

    @torch.no_grad()
    def predict(self, text: str) -> ClassificationResult:
        """Detect spam content."""
        self._ensure_loaded()
        
        if not text.strip():
            return ClassificationResult(
                classifier=self.name,
                flagged=False,
                confidence=0.0,
                severity=Severity.NONE,
            )
        
        # Heuristic scoring
        pattern_matches, heuristic_score = self._check_heuristics(text)
        
        # ML scoring
        ml_score = 0.0
        if self.use_ml_model and self._tokenizer is not None:
            ml_score = self._get_ml_score(text)
        
        # Combine scores
        if self.use_ml_model:
            confidence = 0.6 * ml_score + 0.4 * heuristic_score
        else:
            confidence = heuristic_score
        
        flagged = confidence >= self.threshold
        
        return ClassificationResult(
            classifier=self.name,
            flagged=flagged,
            confidence=confidence,
            severity=self._get_severity(confidence),
            categories=list(pattern_matches.keys()) if flagged else [],
            metadata={
                "pattern_matches": pattern_matches,
                "heuristic_score": round(heuristic_score, 4),
                "ml_score": round(ml_score, 4),
                "model": self._model_path,
            },
        )

    def _check_heuristics(self, text: str) -> Tuple[Dict[str, int], float]:
        """Check text against spam heuristics."""
        matches: Dict[str, int] = {}
        total_score = 0.0
        
        for name, pattern in self._compiled_patterns.items():
            found = pattern.findall(text)
            if found:
                matches[name] = len(found)
                total_score += self.PATTERN_WEIGHTS.get(name, 0.1)
        
        # Additional heuristics
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.5:
            matches["high_caps_ratio"] = 1
            total_score += 0.15
        
        return matches, min(total_score, 1.0)

    def _get_ml_score(self, text: str) -> float:
        """Get spam probability from ML model."""
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self._device)
        
        outputs = self._model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        
        # Assuming [ham, spam] classification
        spam_prob = probs[0, 1].item() if probs.shape[-1] == 2 else probs[0].max().item()
        return spam_prob

    def _get_severity(self, confidence: float) -> Severity:
        """Map confidence to severity."""
        if confidence < self.threshold:
            return Severity.NONE
        elif confidence < 0.6:
            return Severity.LOW
        elif confidence < 0.75:
            return Severity.MEDIUM
        else:
            return Severity.HIGH

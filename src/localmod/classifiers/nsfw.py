"""NSFW (Not Safe For Work) text content detection."""

import re
from typing import List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from localmod.models.base import BaseClassifier, ClassificationResult, Severity


class NSFWClassifier(BaseClassifier):
    """
    Detects sexually explicit or adult content in text.
    
    Uses a fine-tuned model for NSFW text classification with
    false positive filtering for innocuous content.
    """
    
    name = "nsfw"
    version = "1.1.0"
    
    MODEL_NAME = "michellejieli/NSFW_text_classifier"
    
    NSFW_CATEGORIES = [
        "sexual_content",
        "adult_themes",
        "explicit_language",
    ]
    
    # Explicit keywords that strongly indicate NSFW content
    EXPLICIT_KEYWORDS = {
        "naked", "nude", "sex", "porn", "explicit", "erotic",
        "xxx", "nsfw", "adult only", "18+", "fetish",
        "seduce", "intimate", "nudes", "orgasm", "horny",
    }
    
    # False positive filters - innocuous terms the model may wrongly flag
    SAFE_OVERRIDE_PATTERNS = [
        r'\b(puppy|puppies|kitten|kittens|dog|dogs|cat|cats)\b',
        r'\b(baby|babies|child|children|kid|kids)\b',
        r'\b(cute|adorable|sweet|lovely|beautiful)\s+(animal|pet|day)',
        r'\b(weather|programming|coding|work|meeting|office)\b',
        r'\b(hello|hi|hey|good morning|good night)\b',
    ]

    def __init__(
        self,
        device: str = "auto",
        threshold: float = 0.5,
    ):
        super().__init__(device=device, threshold=threshold)
        self._safe_patterns: List[re.Pattern] = []

    def load(self) -> None:
        """Load NSFW detection model."""
        import os
        # Try local models directory first, then fall back to HuggingFace
        local_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "models", "nsfw-classifier"
        )
        model_path = local_path if os.path.exists(local_path) else self.MODEL_NAME
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
        self._model.to(self._device)
        self._model.eval()
        
        # Compile safe patterns
        self._safe_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SAFE_OVERRIDE_PATTERNS
        ]

    def _contains_explicit_keyword(self, text: str) -> bool:
        """Check if text contains explicit NSFW keywords."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.EXPLICIT_KEYWORDS)

    def _is_likely_safe(self, text: str) -> bool:
        """Check if text matches safe override patterns (to filter false positives)."""
        if self._contains_explicit_keyword(text):
            # If text has explicit keywords, don't override even if safe patterns match
            return False
        return any(pattern.search(text) for pattern in self._safe_patterns)

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
        
        # Check for false positive override FIRST
        # If text appears safe (matches safe patterns but no explicit keywords), 
        # reduce confidence significantly
        safe_override = self._is_likely_safe(text)
        
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
        if probs.shape[-1] == 2:
            nsfw_prob = probs[0, 1].item()
        else:
            nsfw_prob = probs[0].max().item()
        
        # Apply safe override - dramatically reduce confidence for likely false positives
        if safe_override and nsfw_prob > 0.5:
            # Model flagged but text appears innocuous - likely false positive
            nsfw_prob = min(nsfw_prob * 0.1, 0.3)  # Cap at 30%
        
        flagged = nsfw_prob >= self.threshold
        
        return ClassificationResult(
            classifier=self.name,
            flagged=flagged,
            confidence=nsfw_prob,
            severity=self._get_severity(nsfw_prob),
            categories=["sexual_content"] if flagged else [],
            metadata={
                "model": self.MODEL_NAME,
                "safe_override_applied": safe_override,
            },
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
        
        # Pre-compute safe overrides
        safe_overrides = [self._is_likely_safe(t) for t in valid_texts]
        
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
            
            # Apply safe override
            if safe_overrides[idx] and nsfw_prob > 0.5:
                nsfw_prob = min(nsfw_prob * 0.1, 0.3)
            
            flagged = nsfw_prob >= self.threshold
            results[orig_idx] = ClassificationResult(
                classifier=self.name,
                flagged=flagged,
                confidence=nsfw_prob,
                severity=self._get_severity(nsfw_prob),
                categories=["sexual_content"] if flagged else [],
                metadata={
                    "model": self.MODEL_NAME,
                    "safe_override_applied": safe_overrides[idx],
                },
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

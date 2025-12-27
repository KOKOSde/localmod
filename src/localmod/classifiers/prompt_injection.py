"""Prompt injection and jailbreak detection for LLM safety."""

import re
from typing import Dict, List, Optional, Pattern

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from localmod.models.base import BaseClassifier, ClassificationResult, Severity


class PromptInjectionDetector(BaseClassifier):
    """
    Detects prompt injection attempts and LLM jailbreaks.
    
    Uses a combination of:
    1. Pattern matching for known injection techniques
    2. ML model for semantic detection
    
    Model: deepset/deberta-v3-base-injection (or similar)
    """
    
    name = "prompt_injection"
    version = "1.0.0"
    
    # Known injection patterns
    INJECTION_PATTERNS: Dict[str, List[str]] = {
        "instruction_override": [
            r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
            r"disregard\s+(all\s+)?(previous|prior|above)",
            r"forget\s+(everything|all)\s+(above|before|prior)",
        ],
        "role_manipulation": [
            r"you\s+are\s+now\s+(?!going|about)",
            r"pretend\s+(to\s+be|you('re|\s+are))",
            r"act\s+as\s+(if\s+you('re|\s+are)|a)",
            r"roleplay\s+as",
            r"from\s+now\s+on,?\s+you('re|\s+are)",
        ],
        "system_prompt_extraction": [
            r"(print|show|display|reveal|output)\s+(your\s+)?(system\s+)?(prompt|instructions)",
            r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions)",
            r"repeat\s+(your\s+)?(initial|system)\s+(prompt|instructions)",
        ],
        "jailbreak_keywords": [
            r"\bDAN\b",  # "Do Anything Now"
            r"\bJailbreak(ed)?\b",
            r"developer\s+mode",
            r"unrestricted\s+mode",
            r"no\s+(restrictions?|limits?|rules?)",
        ],
        "encoding_evasion": [
            r"base64",
            r"rot13",
            r"hex\s*:",
            r"\\x[0-9a-f]{2}",
        ],
        "markdown_escape": [
            r"```[\s\S]*?(ignore|forget|system|admin)",
            r"\[SYSTEM\]",
            r"\[ADMIN\]",
            r"<\|.*?\|>",  # Token markers
        ],
    }
    
    # ML model for semantic detection
    MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"

    def __init__(
        self,
        device: str = "auto",
        threshold: float = 0.5,
        use_ml_model: bool = True,
    ):
        super().__init__(device=device, threshold=threshold)
        self.use_ml_model = use_ml_model
        self._compiled_patterns: Dict[str, List[Pattern]] = {}

    def load(self) -> None:
        """Load detection patterns and optionally the ML model."""
        # Compile regex patterns
        for category, patterns in self.INJECTION_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        # Load ML model if enabled
        if self.use_ml_model:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.MODEL_NAME
                )
                self._model.to(self._device)
                self._model.eval()
            except Exception:
                # Fall back to pattern-only detection
                self.use_ml_model = False
                self._model = True  # Mark as loaded
        else:
            self._model = True  # Mark as loaded

    @torch.no_grad()
    def predict(self, text: str) -> ClassificationResult:
        """Detect prompt injection attempts."""
        self._ensure_loaded()
        
        if not text.strip():
            return ClassificationResult(
                classifier=self.name,
                flagged=False,
                confidence=0.0,
                severity=Severity.NONE,
            )
        
        # Pattern-based detection
        pattern_matches = self._check_patterns(text)
        pattern_score = min(len(pattern_matches) * 0.3, 0.9) if pattern_matches else 0.0
        
        # ML-based detection
        ml_score = 0.0
        if self.use_ml_model and self._tokenizer is not None:
            ml_score = self._get_ml_score(text)
        
        # Combine scores (weighted average, pattern matches boost confidence)
        if self.use_ml_model:
            confidence = max(ml_score, pattern_score)
            if pattern_matches and ml_score > 0.3:
                confidence = min(confidence + 0.1, 1.0)
        else:
            confidence = pattern_score
        
        flagged = confidence >= self.threshold
        
        return ClassificationResult(
            classifier=self.name,
            flagged=flagged,
            confidence=confidence,
            severity=self._get_severity(confidence, pattern_matches),
            categories=list(pattern_matches.keys()) if flagged else [],
            metadata={
                "pattern_matches": {k: len(v) for k, v in pattern_matches.items()},
                "ml_score": round(ml_score, 4),
                "pattern_score": round(pattern_score, 4),
            },
        )

    def _check_patterns(self, text: str) -> Dict[str, List[str]]:
        """Check text against known injection patterns."""
        matches: Dict[str, List[str]] = {}
        
        for category, patterns in self._compiled_patterns.items():
            category_matches: List[str] = []
            for pattern in patterns:
                found = pattern.findall(text)
                if found:
                    if isinstance(found[0], str):
                        category_matches.extend(found)
                    else:
                        category_matches.extend([f[0] for f in found])
            if category_matches:
                matches[category] = category_matches
        
        return matches

    def _get_ml_score(self, text: str) -> float:
        """Get injection probability from ML model."""
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self._device)
        
        outputs = self._model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        
        # Assuming binary classification: [safe, injection]
        injection_prob = probs[0, 1].item() if probs.shape[-1] == 2 else probs[0].max().item()
        return injection_prob

    def _get_severity(self, confidence: float, pattern_matches: Dict) -> Severity:
        """Determine severity based on detection results."""
        if confidence < self.threshold:
            return Severity.NONE
        
        # System prompt extraction or instruction override is critical
        critical_categories = {"system_prompt_extraction", "instruction_override"}
        if pattern_matches.keys() & critical_categories:
            return Severity.CRITICAL
        
        if confidence >= 0.85:
            return Severity.CRITICAL
        elif confidence >= 0.7:
            return Severity.HIGH
        elif confidence >= 0.5:
            return Severity.MEDIUM
        else:
            return Severity.LOW


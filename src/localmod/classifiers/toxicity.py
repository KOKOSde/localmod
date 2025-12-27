"""Toxicity detection classifier using transformer models."""

from typing import List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from localmod.models.base import BaseClassifier, ClassificationResult, Severity


class ToxicityClassifier(BaseClassifier):
    """
    Detects toxic content including hate speech, harassment, threats, and profanity.
    
    Uses a fine-tuned DistilBERT model for efficient inference.
    Model: martin-ha/toxic-comment-model (or similar)
    """
    
    name = "toxicity"
    version = "1.0.0"
    
    # Model options (in order of preference)
    # unitary/toxic-bert is more accurate for general toxicity detection
    MODEL_OPTIONS = [
        "unitary/toxic-bert",  # Multi-label, better accuracy
        "martin-ha/toxic-comment-model",  # Binary, faster but less accurate
        "s-nlp/roberta_toxicity_classifier",
    ]
    
    TOXICITY_CATEGORIES = [
        "toxic",
        "severe_toxic", 
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        threshold: float = 0.5,
    ):
        super().__init__(device=device, threshold=threshold)
        self.model_name = model_name or self.MODEL_OPTIONS[0]

    def load(self) -> None:
        """Load the toxicity detection model."""
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> ClassificationResult:
        """Classify text for toxicity."""
        self._ensure_loaded()
        
        # Handle empty or whitespace text
        if not text.strip():
            return ClassificationResult(
                classifier=self.name,
                flagged=False,
                confidence=0.0,
                severity=Severity.NONE,
                explanation="Empty text",
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
        
        # Handle multi-label vs binary classification
        num_labels = outputs.logits.shape[-1]
        if num_labels > 2:
            # Multi-label model (e.g., unitary/toxic-bert): use sigmoid
            probabilities = torch.sigmoid(outputs.logits)
            toxic_prob = probabilities[0].max().item()
        else:
            # Binary classification: use softmax
            probabilities = torch.softmax(outputs.logits, dim=-1)
            toxic_prob = probabilities[0, 1].item()
        
        flagged = toxic_prob >= self.threshold
        severity = self._get_severity(toxic_prob)
        categories = self._get_categories(probabilities[0]) if flagged else []
        
        return ClassificationResult(
            classifier=self.name,
            flagged=flagged,
            confidence=toxic_prob,
            severity=severity,
            categories=categories,
            metadata={"model": self.model_name},
        )

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Batch prediction for efficiency."""
        self._ensure_loaded()
        
        if not texts:
            return []
        
        # Filter empty texts
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
        probabilities = torch.softmax(outputs.logits, dim=-1)
        
        # Build results
        results: List[Optional[ClassificationResult]] = [None] * len(texts)
        
        for idx, orig_idx in enumerate(valid_indices):
            if probabilities.shape[-1] == 2:
                toxic_prob = probabilities[idx, 1].item()
            else:
                toxic_prob = probabilities[idx].max().item()
            
            flagged = toxic_prob >= self.threshold
            results[orig_idx] = ClassificationResult(
                classifier=self.name,
                flagged=flagged,
                confidence=toxic_prob,
                severity=self._get_severity(toxic_prob),
                categories=self._get_categories(probabilities[idx]) if flagged else [],
                metadata={"model": self.model_name},
            )
        
        # Fill empty text results
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
        """Map confidence score to severity level."""
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

    def _get_categories(self, probs: torch.Tensor) -> List[str]:
        """Extract which toxicity categories were detected."""
        categories = []
        if len(probs) > 2:
            # Multi-label model
            for i, cat in enumerate(self.TOXICITY_CATEGORIES[:len(probs)]):
                if probs[i].item() >= self.threshold:
                    categories.append(cat)
        elif probs[-1].item() >= self.threshold:
            categories.append("toxic")
        return categories


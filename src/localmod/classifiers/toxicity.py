"""Toxicity detection classifier using weighted ensemble of transformer models.

This classifier achieves 0.75 balanced accuracy on CHI 2025 benchmark datasets,
matching Amazon Comprehend and outperforming Perspective API and Google NL.

Ensemble composition:
- 50% Unitary Toxic-BERT (unitary/toxic-bert)
- 20% DeHateBERT (Hate-speech-CNERG/dehatebert-mono-english)
- 15% s-nlp RoBERTa (s-nlp/roberta_toxicity_classifier)
- 15% Facebook Dynabench (facebook/roberta-hate-speech-dynabench-r4-target)
"""

from typing import List, Optional, Dict, Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from localmod.models.base import BaseClassifier, ClassificationResult, Severity
from localmod.models.paths import (
    get_classifier_model_path,
    get_transformers_kwargs,
    TOXICITY_ENSEMBLE_MODELS,
    TOXICITY_ENSEMBLE_WEIGHTS,
    model_exists_locally,
)


class ToxicityClassifier(BaseClassifier):
    """
    Detects toxic content using a weighted ensemble of transformer models.
    
    Achieves 0.75 balanced accuracy on CHI 2025 benchmark (HateXplain, 
    Civil Comments, SBIC), matching Amazon Comprehend.
    
    Models used:
    - Unitary Toxic-BERT (50% weight) - multi-label toxicity
    - DeHateBERT (20% weight) - hate speech detection
    - s-nlp RoBERTa (15% weight) - toxicity classification
    - Facebook Dynabench (15% weight) - adversarially robust hate speech
    """
    
    name = "toxicity"
    version = "2.0.0"  # Ensemble version
    
    # Optimal threshold from CHI 2025 benchmark testing
    DEFAULT_THRESHOLD = 0.17
    
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
        device: str = "auto",
        threshold: float = None,
        use_ensemble: bool = True,
    ):
        # Use optimized threshold by default
        if threshold is None:
            threshold = self.DEFAULT_THRESHOLD
        super().__init__(device=device, threshold=threshold)
        
        self._use_ensemble = use_ensemble
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._model_types: Dict[str, str] = {}

    def load(self) -> None:
        """Load the toxicity detection model(s)."""
        kwargs = get_transformers_kwargs()
        
        if self._use_ensemble:
            self._load_ensemble(kwargs)
        else:
            self._load_single_model(kwargs)

    def _load_single_model(self, kwargs: Dict[str, Any]) -> None:
        """Load only the primary toxicity model."""
        model_path = get_classifier_model_path("toxicity")
        self._tokenizers["toxicity"] = AutoTokenizer.from_pretrained(model_path, **kwargs)
        self._models["toxicity"] = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
        self._models["toxicity"].to(self._device)
        self._models["toxicity"].eval()
        self._model_types["toxicity"] = "multilabel"

    def _load_ensemble(self, kwargs: Dict[str, Any]) -> None:
        """Load all ensemble models."""
        model_type_map = {
            "toxicity": "multilabel",
            "toxicity_dehatebert": "binary",
            "toxicity_snlp": "binary",
            "toxicity_facebook": "binary",
        }
        
        for model_name in TOXICITY_ENSEMBLE_MODELS:
            # Skip if model doesn't exist locally and we're not using fallback
            if not model_exists_locally(model_name):
                # Try to load anyway - might download from HF
                pass
            
            try:
                model_path = get_classifier_model_path(model_name)
                self._tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path, **kwargs)
                self._models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
                self._models[model_name].to(self._device)
                self._models[model_name].eval()
                self._model_types[model_name] = model_type_map.get(model_name, "binary")
            except Exception as e:
                # Log but continue - ensemble can work with fewer models
                print(f"Warning: Could not load {model_name}: {e}")
        
        if not self._models:
            raise RuntimeError("No toxicity models could be loaded")

    def _get_toxicity_prob(self, model_name: str, inputs: Dict[str, torch.Tensor]) -> float:
        """Get toxicity probability from a single model."""
        outputs = self._models[model_name](**inputs)
        
        if self._model_types[model_name] == "multilabel":
            # Multi-label: use sigmoid and take max
            probs = torch.sigmoid(outputs.logits)
            return probs[0].max().item()
        else:
            # Binary: use softmax, toxic is index 1
            probs = torch.softmax(outputs.logits, dim=-1)
            return probs[0, 1].item()

    @torch.no_grad()
    def predict(self, text: str) -> ClassificationResult:
        """Classify text for toxicity using weighted ensemble."""
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
        
        # Compute weighted ensemble probability
        weighted_prob = 0.0
        total_weight = 0.0
        model_probs = {}
        
        for model_name, model in self._models.items():
            tokenizer = self._tokenizers[model_name]
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            prob = self._get_toxicity_prob(model_name, inputs)
            model_probs[model_name] = prob
            
            weight = TOXICITY_ENSEMBLE_WEIGHTS.get(model_name, 0.25)
            weighted_prob += prob * weight
            total_weight += weight
        
        # Normalize if not all models loaded
        if total_weight > 0 and total_weight < 1.0:
            weighted_prob /= total_weight
        
        flagged = weighted_prob >= self.threshold
        severity = self._get_severity(weighted_prob)
        
        return ClassificationResult(
            classifier=self.name,
            flagged=flagged,
            confidence=weighted_prob,
            severity=severity,
            categories=["toxic"] if flagged else [],
            metadata={
                "ensemble_models": list(self._models.keys()),
                "model_probs": model_probs,
                "threshold": self.threshold,
            },
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
        
        # Get probabilities from each model
        all_probs = {name: [] for name in self._models.keys()}
        
        for model_name in self._models.keys():
            tokenizer = self._tokenizers[model_name]
            model = self._models[model_name]
            
            # Batch tokenize
            inputs = tokenizer(
                valid_texts,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            if self._model_types[model_name] == "multilabel":
                probs = torch.sigmoid(outputs.logits).max(dim=-1).values
            else:
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            
            all_probs[model_name] = probs.cpu().tolist()
        
        # Compute weighted ensemble for each text
        results: List[Optional[ClassificationResult]] = [None] * len(texts)
        
        for idx, orig_idx in enumerate(valid_indices):
            weighted_prob = 0.0
            total_weight = 0.0
            
            for model_name in self._models.keys():
                weight = TOXICITY_ENSEMBLE_WEIGHTS.get(model_name, 0.25)
                weighted_prob += all_probs[model_name][idx] * weight
                total_weight += weight
            
            if total_weight > 0 and total_weight < 1.0:
                weighted_prob /= total_weight
            
            flagged = weighted_prob >= self.threshold
            
            results[orig_idx] = ClassificationResult(
                classifier=self.name,
                flagged=flagged,
                confidence=weighted_prob,
                severity=self._get_severity(weighted_prob),
                categories=["toxic"] if flagged else [],
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
        elif confidence < 0.4:
            return Severity.LOW
        elif confidence < 0.6:
            return Severity.MEDIUM
        elif confidence < 0.8:
            return Severity.HIGH
        else:
            return Severity.CRITICAL

    def _get_categories(self, probs: torch.Tensor) -> List[str]:
        """Extract which toxicity categories were detected."""
        categories = []
        if len(probs) > 2:
            for i, cat in enumerate(self.TOXICITY_CATEGORIES[:len(probs)]):
                if probs[i].item() >= self.threshold:
                    categories.append(cat)
        elif probs[-1].item() >= self.threshold:
            categories.append("toxic")
        return categories

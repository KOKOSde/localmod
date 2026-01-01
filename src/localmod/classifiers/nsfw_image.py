"""NSFW image content detection using Falconsai/nsfw_image_detection.

This classifier uses a Vision Transformer (ViT) model fine-tuned for NSFW detection.
- 71M+ downloads on HuggingFace (most popular NSFW image model)
- Apache 2.0 license (commercial use OK)
- Binary classification: nsfw / normal
"""

import os
from typing import List, Optional, Union, BinaryIO
from pathlib import Path

import torch
from PIL import Image

from localmod.models.base import BaseClassifier, ClassificationResult, Severity


class ImageNSFWClassifier(BaseClassifier):
    """
    Detects NSFW (Not Safe For Work) content in images.
    
    Uses Falconsai/nsfw_image_detection - a ViT-based model with 71M+ downloads.
    
    Supports:
    - File paths (str or Path)
    - PIL Image objects
    - Bytes/BytesIO
    - URLs (downloads and analyzes)
    """
    
    name = "nsfw_image"
    version = "1.0.0"
    
    MODEL_ID = "Falconsai/nsfw_image_detection"
    
    NSFW_CATEGORIES = [
        "nsfw",
        "explicit",
        "adult_content",
    ]

    def __init__(
        self,
        device: str = "auto",
        threshold: float = 0.5,
    ):
        super().__init__(device=device, threshold=threshold)
        self._processor = None
        self._model = None

    def load(self) -> None:
        """Load the NSFW image detection model."""
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        
        # Check for local model first
        model_dir = os.environ.get("LOCALMOD_MODEL_DIR", "~/.cache/localmod/models")
        model_dir = os.path.expanduser(model_dir)
        local_path = os.path.join(model_dir, "nsfw_image")
        
        if os.path.exists(local_path) and os.path.isdir(local_path):
            model_path = local_path
        else:
            model_path = self.MODEL_ID
        
        # Check offline mode
        offline = os.environ.get("LOCALMOD_OFFLINE", "").lower() in ("1", "true", "yes")
        kwargs = {"local_files_only": True} if offline else {}
        
        self._processor = AutoImageProcessor.from_pretrained(model_path, **kwargs)
        self._model = AutoModelForImageClassification.from_pretrained(model_path, **kwargs)
        self._model.to(self._device)
        self._model.eval()

    def _load_image(self, image_input: Union[str, Path, bytes, BinaryIO, Image.Image]) -> Image.Image:
        """Load image from various input types."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        
        if isinstance(image_input, (str, Path)):
            path = str(image_input)
            
            # Check if it's a URL
            if path.startswith(("http://", "https://")):
                import requests
                from io import BytesIO
                response = requests.get(path, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            
            # Local file
            return Image.open(path).convert("RGB")
        
        if isinstance(image_input, bytes):
            from io import BytesIO
            return Image.open(BytesIO(image_input)).convert("RGB")
        
        if hasattr(image_input, "read"):
            return Image.open(image_input).convert("RGB")
        
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    @torch.no_grad()
    def predict(self, image_input: Union[str, Path, bytes, BinaryIO, Image.Image]) -> ClassificationResult:
        """
        Detect NSFW content in an image.
        
        Args:
            image_input: Can be:
                - File path (str or Path)
                - URL (str starting with http:// or https://)
                - Raw bytes
                - File-like object (BytesIO)
                - PIL Image object
        
        Returns:
            ClassificationResult with flagged=True if NSFW detected
        """
        self._ensure_loaded()
        
        try:
            image = self._load_image(image_input)
        except Exception as e:
            return ClassificationResult(
                classifier=self.name,
                flagged=False,
                confidence=0.0,
                severity=Severity.NONE,
                explanation=f"Failed to load image: {e}",
                metadata={"error": str(e)},
            )
        
        # Process image
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Inference
        outputs = self._model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        
        # Get label mapping
        id2label = self._model.config.id2label
        
        # Find NSFW probability
        nsfw_prob = 0.0
        predicted_label = ""
        
        for idx, prob in enumerate(probs[0]):
            label = id2label.get(idx, f"label_{idx}").lower()
            if "nsfw" in label or "porn" in label or "explicit" in label:
                nsfw_prob = max(nsfw_prob, prob.item())
            if prob.item() == probs[0].max().item():
                predicted_label = label
        
        # If no explicit NSFW label found, use inverse of "normal" probability
        if nsfw_prob == 0.0 and "normal" in predicted_label.lower():
            nsfw_prob = 1.0 - probs[0].max().item()
        elif nsfw_prob == 0.0:
            # Model predicts NSFW as highest class
            nsfw_prob = probs[0].max().item() if "nsfw" in predicted_label.lower() else 0.0
        
        flagged = nsfw_prob >= self.threshold
        
        return ClassificationResult(
            classifier=self.name,
            flagged=flagged,
            confidence=nsfw_prob,
            severity=self._get_severity(nsfw_prob),
            categories=["nsfw"] if flagged else [],
            metadata={
                "model": self.MODEL_ID,
                "predicted_label": predicted_label,
                "all_probs": {id2label.get(i, f"label_{i}"): round(p.item(), 4) for i, p in enumerate(probs[0])},
            },
        )

    @torch.no_grad()
    def predict_batch(self, image_inputs: List[Union[str, Path, bytes, Image.Image]]) -> List[ClassificationResult]:
        """Batch prediction for multiple images."""
        self._ensure_loaded()
        
        if not image_inputs:
            return []
        
        # Load all images
        images = []
        errors = {}
        
        for i, img_input in enumerate(image_inputs):
            try:
                images.append(self._load_image(img_input))
            except Exception as e:
                images.append(None)
                errors[i] = str(e)
        
        # Filter valid images
        valid_indices = [i for i, img in enumerate(images) if img is not None]
        valid_images = [images[i] for i in valid_indices]
        
        if not valid_images:
            return [
                ClassificationResult(
                    classifier=self.name,
                    flagged=False,
                    confidence=0.0,
                    severity=Severity.NONE,
                    explanation=f"Failed to load image: {errors.get(i, 'Unknown error')}",
                )
                for i in range(len(image_inputs))
            ]
        
        # Batch process
        inputs = self._processor(images=valid_images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        outputs = self._model(**inputs)
        all_probs = torch.softmax(outputs.logits, dim=-1)
        
        id2label = self._model.config.id2label
        
        # Build results
        results: List[Optional[ClassificationResult]] = [None] * len(image_inputs)
        
        for idx, orig_idx in enumerate(valid_indices):
            probs = all_probs[idx]
            
            nsfw_prob = 0.0
            predicted_label = ""
            
            for label_idx, prob in enumerate(probs):
                label = id2label.get(label_idx, f"label_{label_idx}").lower()
                if "nsfw" in label or "porn" in label or "explicit" in label:
                    nsfw_prob = max(nsfw_prob, prob.item())
                if prob.item() == probs.max().item():
                    predicted_label = label
            
            if nsfw_prob == 0.0 and "normal" in predicted_label.lower():
                nsfw_prob = 1.0 - probs.max().item()
            elif nsfw_prob == 0.0:
                nsfw_prob = probs.max().item() if "nsfw" in predicted_label.lower() else 0.0
            
            flagged = nsfw_prob >= self.threshold
            
            results[orig_idx] = ClassificationResult(
                classifier=self.name,
                flagged=flagged,
                confidence=nsfw_prob,
                severity=self._get_severity(nsfw_prob),
                categories=["nsfw"] if flagged else [],
                metadata={
                    "model": self.MODEL_ID,
                    "predicted_label": predicted_label,
                },
            )
        
        # Fill error results
        for i, result in enumerate(results):
            if result is None:
                results[i] = ClassificationResult(
                    classifier=self.name,
                    flagged=False,
                    confidence=0.0,
                    severity=Severity.NONE,
                    explanation=f"Failed to load image: {errors.get(i, 'Unknown error')}",
                )
        
        return results  # type: ignore

    def _get_severity(self, confidence: float) -> Severity:
        """Map confidence to severity level."""
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

    # Override text-based predict to raise helpful error
    def predict_text(self, text: str) -> ClassificationResult:
        """This classifier is for images only. Use predict() with an image input."""
        raise NotImplementedError(
            f"{self.name} is an image classifier. "
            "Use predict(image_path) or predict(pil_image) instead of predict(text)."
        )


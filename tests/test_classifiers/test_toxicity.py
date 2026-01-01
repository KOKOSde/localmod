"""Tests for toxicity classifier."""

import pytest
from localmod.classifiers.toxicity import ToxicityClassifier
from localmod.models.base import Severity


class TestToxicityClassifier:
    """Tests for ToxicityClassifier class."""

    def test_classifier_attributes(self):
        """Test classifier has correct attributes."""
        classifier = ToxicityClassifier()
        
        assert classifier.name == "toxicity"
        assert classifier.version == "2.0.0"  # Ensemble version
        assert classifier.threshold == 0.17  # Optimized threshold from CHI 2025 benchmark

    def test_custom_threshold(self):
        """Test classifier with custom threshold."""
        classifier = ToxicityClassifier(threshold=0.8)
        
        assert classifier.threshold == 0.8

    def test_not_loaded_initially(self):
        """Test classifier is not loaded on initialization."""
        classifier = ToxicityClassifier()
        
        assert classifier.is_loaded is False

    def test_severity_mapping(self):
        """Test severity level mapping."""
        classifier = ToxicityClassifier(threshold=0.17)
        
        # Thresholds: <0.17=NONE, <0.4=LOW, <0.6=MEDIUM, <0.8=HIGH, >=0.8=CRITICAL
        assert classifier._get_severity(0.1) == Severity.NONE
        assert classifier._get_severity(0.3) == Severity.LOW
        assert classifier._get_severity(0.5) == Severity.MEDIUM
        assert classifier._get_severity(0.7) == Severity.HIGH
        assert classifier._get_severity(0.9) == Severity.CRITICAL


@pytest.mark.slow
class TestToxicityClassifierWithModel:
    """Tests that require loading the ML model (marked as slow)."""

    @pytest.fixture
    def loaded_classifier(self):
        """Create and load a toxicity classifier."""
        classifier = ToxicityClassifier(device="cpu")
        classifier.load()
        return classifier

    def test_load_model(self, loaded_classifier):
        """Test that model loads successfully."""
        assert loaded_classifier.is_loaded is True
        assert loaded_classifier._model is not None
        assert loaded_classifier._tokenizer is not None

    def test_predict_safe_text(self, loaded_classifier):
        """Test prediction on safe text."""
        result = loaded_classifier.predict("Hello, how are you today?")
        
        assert result.classifier == "toxicity"
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_empty_text(self, loaded_classifier):
        """Test prediction on empty text."""
        result = loaded_classifier.predict("")
        
        assert result.flagged is False
        assert result.confidence == 0.0
        assert result.severity == Severity.NONE

    def test_predict_batch_empty(self, loaded_classifier):
        """Test batch prediction with empty list."""
        results = loaded_classifier.predict_batch([])
        
        assert results == []

    def test_predict_batch_mixed(self, loaded_classifier):
        """Test batch prediction with mixed content."""
        texts = [
            "Hello, world!",
            "",
            "This is a test.",
        ]
        results = loaded_classifier.predict_batch(texts)
        
        assert len(results) == 3
        assert results[1].flagged is False  # Empty text



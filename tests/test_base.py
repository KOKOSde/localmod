"""Tests for base classifier module."""

import pytest
from localmod.models.base import BaseClassifier, ClassificationResult, Severity


class TestSeverity:
    """Tests for Severity enum."""

    def test_severity_values(self):
        """Test that all severity levels exist with correct values."""
        assert Severity.NONE.value == "none"
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"
        assert Severity.CRITICAL.value == "critical"

    def test_severity_ordering(self):
        """Test that severity levels can be compared via list index."""
        levels = list(Severity)
        assert levels.index(Severity.NONE) < levels.index(Severity.LOW)
        assert levels.index(Severity.LOW) < levels.index(Severity.MEDIUM)
        assert levels.index(Severity.MEDIUM) < levels.index(Severity.HIGH)
        assert levels.index(Severity.HIGH) < levels.index(Severity.CRITICAL)


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_basic_creation(self):
        """Test creating a basic result."""
        result = ClassificationResult(
            classifier="test",
            flagged=False,
            confidence=0.1,
            severity=Severity.NONE,
        )
        assert result.classifier == "test"
        assert result.flagged is False
        assert result.confidence == 0.1
        assert result.severity == Severity.NONE
        assert result.categories == []
        assert result.metadata == {}
        assert result.explanation == ""

    def test_full_creation(self):
        """Test creating a result with all fields."""
        result = ClassificationResult(
            classifier="toxicity",
            flagged=True,
            confidence=0.95,
            severity=Severity.HIGH,
            categories=["hate", "harassment"],
            metadata={"model": "test-model"},
            explanation="Content flagged for toxicity",
        )
        assert result.categories == ["hate", "harassment"]
        assert result.metadata == {"model": "test-model"}
        assert result.explanation == "Content flagged for toxicity"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = ClassificationResult(
            classifier="test",
            flagged=True,
            confidence=0.12345678,
            severity=Severity.MEDIUM,
            categories=["spam"],
            metadata={"key": "value"},
            explanation="Test explanation",
        )
        d = result.to_dict()
        
        assert d["classifier"] == "test"
        assert d["flagged"] is True
        assert d["confidence"] == 0.1235  # Rounded to 4 decimal places
        assert d["severity"] == "medium"  # String value
        assert d["categories"] == ["spam"]
        assert d["metadata"] == {"key": "value"}
        assert d["explanation"] == "Test explanation"


class TestBaseClassifier:
    """Tests for BaseClassifier abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseClassifier cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseClassifier()  # type: ignore

    def test_device_resolution_cpu(self):
        """Test device resolution returns valid device."""
        device = BaseClassifier._resolve_device("cpu")
        assert device == "cpu"

    def test_device_resolution_cuda(self):
        """Test device resolution for cuda."""
        device = BaseClassifier._resolve_device("cuda")
        assert device == "cuda"

    def test_device_resolution_auto(self):
        """Test auto device resolution."""
        device = BaseClassifier._resolve_device("auto")
        assert device in ["cpu", "cuda"]

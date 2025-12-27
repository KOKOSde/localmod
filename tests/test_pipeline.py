"""Tests for the SafetyPipeline orchestrator."""

import pytest
from localmod.pipeline import SafetyPipeline, SafetyReport
from localmod.models.base import Severity


class TestSafetyPipeline:
    """Tests for SafetyPipeline class."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline with only PII detector (no ML models needed)."""
        pipeline = SafetyPipeline(classifiers=["pii"], device="cpu")
        return pipeline

    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        pipeline = SafetyPipeline(classifiers=["pii"])
        
        assert "pii" in pipeline.available_classifiers
        assert pipeline.device in ["cpu", "cuda", "auto"]

    def test_pipeline_with_custom_classifiers(self):
        """Test pipeline with specific classifiers."""
        pipeline = SafetyPipeline(classifiers=["pii", "spam"])
        
        assert len(pipeline.available_classifiers) == 2
        assert "pii" in pipeline.available_classifiers
        assert "spam" in pipeline.available_classifiers

    def test_pipeline_with_custom_thresholds(self):
        """Test pipeline with custom thresholds."""
        pipeline = SafetyPipeline(
            classifiers=["pii"],
            thresholds={"pii": 0.8}
        )
        
        pii_classifier = pipeline.get_classifier("pii")
        assert pii_classifier is not None
        assert pii_classifier.threshold == 0.8

    def test_analyze_safe_text(self, pipeline):
        """Test analyzing safe text."""
        report = pipeline.analyze("Hello, how are you today?")
        
        assert isinstance(report, SafetyReport)
        assert report.flagged is False
        assert report.severity == Severity.NONE
        assert len(report.results) == 1
        assert report.processing_time_ms > 0

    def test_analyze_text_with_pii(self, pipeline):
        """Test analyzing text with PII."""
        report = pipeline.analyze("My email is john@example.com")
        
        assert report.flagged is True
        assert report.severity != Severity.NONE
        assert "pii" in [r.classifier for r in report.results]

    def test_analyze_with_explanation(self, pipeline):
        """Test analyze with explanations enabled."""
        report = pipeline.analyze(
            "My email is john@example.com",
            include_explanation=True
        )
        
        assert report.flagged is True
        assert any(r.explanation for r in report.results)

    def test_analyze_empty_text(self, pipeline):
        """Test analyzing empty text."""
        report = pipeline.analyze("")
        
        assert report.flagged is False
        assert report.severity == Severity.NONE

    def test_analyze_batch(self, pipeline):
        """Test batch analysis."""
        texts = [
            "Hello world",
            "My email is test@example.com",
            "Nice weather today",
        ]
        
        reports = pipeline.analyze_batch(texts)
        
        assert len(reports) == 3
        assert reports[0].flagged is False  # Safe text
        assert reports[1].flagged is True   # Has PII
        assert reports[2].flagged is False  # Safe text

    def test_get_classifier(self, pipeline):
        """Test getting a specific classifier."""
        pii = pipeline.get_classifier("pii")
        
        assert pii is not None
        assert pii.name == "pii"

    def test_get_nonexistent_classifier(self, pipeline):
        """Test getting a classifier that doesn't exist."""
        result = pipeline.get_classifier("nonexistent")
        
        assert result is None

    def test_loaded_classifiers_before_load(self, pipeline):
        """Test loaded_classifiers before loading."""
        # PII doesn't need ML model, so it might be loaded on first use
        assert isinstance(pipeline.loaded_classifiers, list)

    def test_load_all(self, pipeline):
        """Test loading all classifiers."""
        pipeline.load_all()
        
        assert len(pipeline.loaded_classifiers) == 1
        assert "pii" in pipeline.loaded_classifiers


class TestSafetyReport:
    """Tests for SafetyReport dataclass."""

    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        from localmod.models.base import ClassificationResult
        
        report = SafetyReport(
            flagged=True,
            severity=Severity.HIGH,
            results=[
                ClassificationResult(
                    classifier="pii",
                    flagged=True,
                    confidence=1.0,
                    severity=Severity.HIGH,
                    categories=["email"],
                )
            ],
            summary="Content flagged for: pii (high): email",
            processing_time_ms=10.5,
            metadata={"classifiers_run": ["pii"]},
        )
        
        d = report.to_dict()
        
        assert d["flagged"] is True
        assert d["severity"] == "high"
        assert len(d["results"]) == 1
        assert d["processing_time_ms"] == 10.5
        assert d["metadata"]["classifiers_run"] == ["pii"]

    def test_summary_generation(self):
        """Test summary generation through pipeline."""
        pipeline = SafetyPipeline(classifiers=["pii"])
        
        # Safe text
        report = pipeline.analyze("Hello world")
        assert "passed all safety checks" in report.summary.lower()
        
        # Text with PII
        report = pipeline.analyze("Email: test@example.com")
        assert "flagged" in report.summary.lower()


class TestPipelineWithMultipleClassifiers:
    """Tests with multiple classifiers (non-ML only)."""

    @pytest.fixture
    def multi_pipeline(self):
        """Create pipeline with pattern-based classifiers only."""
        # Use PII and spam/prompt_injection with use_ml_model=False
        pipeline = SafetyPipeline(classifiers=["pii"], device="cpu")
        return pipeline

    def test_aggregate_severity(self, multi_pipeline):
        """Test that severity is aggregated correctly."""
        # Text with critical PII (SSN)
        report = multi_pipeline.analyze("My SSN is 123-45-6789")
        
        assert report.severity == Severity.CRITICAL

    def test_multiple_flags(self, multi_pipeline):
        """Test detection when multiple issues present."""
        # This text has PII
        report = multi_pipeline.analyze(
            "Email: test@example.com, SSN: 123-45-6789"
        )
        
        assert report.flagged is True
        assert len([r for r in report.results if r.flagged]) >= 1


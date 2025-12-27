"""Integration tests for the full LocalMod pipeline and API."""

import pytest
from fastapi.testclient import TestClient

from localmod import SafetyPipeline, SafetyReport
from localmod.api.app import create_app, reset_pipeline
from localmod.models.base import Severity


class TestFullPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline with PII-only for fast testing."""
        return SafetyPipeline(classifiers=["pii"], device="cpu")

    def test_end_to_end_safe_content(self, pipeline):
        """Test complete flow with safe content."""
        text = "Hello, this is a friendly message about the weather."
        
        report = pipeline.analyze(text, include_explanation=True)
        
        assert isinstance(report, SafetyReport)
        assert report.flagged is False
        assert report.severity == Severity.NONE
        assert "passed all safety checks" in report.summary.lower()
        assert report.processing_time_ms > 0

    def test_end_to_end_flagged_content(self, pipeline):
        """Test complete flow with flagged content."""
        text = "Contact me at john.doe@example.com or call 555-123-4567"
        
        report = pipeline.analyze(text, include_explanation=True)
        
        assert report.flagged is True
        assert report.severity != Severity.NONE
        assert "flagged" in report.summary.lower()
        
        # Check result details
        pii_result = next((r for r in report.results if r.classifier == "pii"), None)
        assert pii_result is not None
        assert pii_result.flagged is True
        assert len(pii_result.categories) >= 1

    def test_batch_processing_integration(self, pipeline):
        """Test batch processing with mixed content."""
        texts = [
            "Hello world",
            "My SSN is 123-45-6789",
            "Nice weather today",
            "Email: test@example.com, Phone: 555-123-4567",
            "Just a normal message",
        ]
        
        reports = pipeline.analyze_batch(texts)
        
        assert len(reports) == 5
        assert reports[0].flagged is False  # Safe
        assert reports[1].flagged is True   # SSN
        assert reports[2].flagged is False  # Safe
        assert reports[3].flagged is True   # Email + Phone
        assert reports[4].flagged is False  # Safe
        
        # Check severity ordering
        assert reports[1].severity == Severity.CRITICAL  # SSN is critical

    def test_report_serialization(self, pipeline):
        """Test that reports serialize correctly."""
        report = pipeline.analyze("My email is test@example.com")
        
        data = report.to_dict()
        
        assert isinstance(data, dict)
        assert "flagged" in data
        assert "severity" in data
        assert "results" in data
        assert "summary" in data
        assert "processing_time_ms" in data
        assert "metadata" in data
        
        # Check nested serialization
        assert isinstance(data["results"], list)
        if data["results"]:
            result = data["results"][0]
            assert "classifier" in result
            assert "confidence" in result
            assert "categories" in result


class TestAPIIntegration:
    """Integration tests for the API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        reset_pipeline()
        app = create_app()
        with TestClient(app) as client:
            yield client
        reset_pipeline()

    def test_full_analysis_workflow(self, client):
        """Test complete API analysis workflow."""
        # 1. Check health
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
        # 2. List classifiers
        response = client.get("/classifiers")
        assert response.status_code == 200
        classifiers = response.json()
        assert "pii" in classifiers
        
        # 3. Analyze safe text
        response = client.post(
            "/analyze",
            json={"text": "Hello world", "classifiers": ["pii"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["flagged"] is False
        
        # 4. Analyze text with PII
        response = client.post(
            "/analyze",
            json={"text": "Email: test@example.com", "classifiers": ["pii"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["flagged"] is True
        
        # 5. Redact PII
        response = client.post(
            "/redact",
            json={"text": "Email: test@example.com"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "test@example.com" not in data["redacted_text"]

    def test_batch_api_workflow(self, client):
        """Test batch API processing."""
        response = client.post(
            "/analyze/batch",
            json={
                "texts": [
                    "Hello",
                    "My email is test@example.com",
                    "World"
                ],
                "classifiers": ["pii"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["results"]) == 3
        assert data["total_flagged"] == 1
        assert data["results"][1]["flagged"] is True

    def test_error_handling(self, client):
        """Test API error handling."""
        # Empty text
        response = client.post(
            "/analyze",
            json={"text": ""}
        )
        assert response.status_code == 422
        
        # Whitespace text
        response = client.post(
            "/analyze",
            json={"text": "   "}
        )
        assert response.status_code == 422

    def test_response_format_consistency(self, client):
        """Test that response formats are consistent."""
        # Single analysis
        response = client.post(
            "/analyze",
            json={"text": "Test", "classifiers": ["pii"]}
        )
        single = response.json()
        
        # Batch analysis
        response = client.post(
            "/analyze/batch",
            json={"texts": ["Test"], "classifiers": ["pii"]}
        )
        batch = response.json()
        
        # Compare formats
        assert set(single.keys()) == set(batch["results"][0].keys())


class TestClassifierIntegration:
    """Integration tests for classifier interactions."""

    def test_pii_redaction_preserves_message_structure(self):
        """Test that PII redaction maintains message readability."""
        from localmod.classifiers.pii import PIIDetector
        
        detector = PIIDetector()
        detector.load()
        
        original = "Contact John at john@example.com or 555-123-4567 for help."
        redacted, detections = detector.redact(original)
        
        # Should have redacted 2 items
        assert len(detections) == 2
        
        # Message should still be readable
        assert "Contact John at" in redacted
        assert "for help." in redacted
        
        # Original PII should be gone
        assert "john@example.com" not in redacted
        assert "555-123-4567" not in redacted

    def test_multiple_pii_types_detected(self):
        """Test detection of multiple PII types in single text."""
        from localmod.classifiers.pii import PIIDetector
        
        detector = PIIDetector()
        detector.load()
        
        text = """
        Name: John Doe
        Email: john.doe@example.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        IP: 192.168.1.1
        """
        
        result = detector.predict(text)
        
        assert result.flagged is True
        assert result.severity == Severity.CRITICAL  # Due to SSN
        
        # Should detect multiple types
        assert len(result.categories) >= 3
        assert "email" in result.categories
        assert "ssn" in result.categories


class TestConfigurationIntegration:
    """Tests for configuration handling."""

    def test_custom_thresholds_affect_results(self):
        """Test that custom thresholds change classification behavior."""
        from localmod.classifiers.pii import PIIDetector
        
        # Default threshold
        detector_default = PIIDetector(threshold=0.5)
        detector_default.load()
        
        # High threshold (stricter)
        detector_strict = PIIDetector(threshold=0.99)
        detector_strict.load()
        
        text = "Contact: test@example.com"
        
        # Both should detect (PII is deterministic with 1.0 confidence)
        result_default = detector_default.predict(text)
        result_strict = detector_strict.predict(text)
        
        # PII detector returns 1.0 confidence for matches
        assert result_default.flagged is True
        assert result_strict.flagged is True  # 1.0 >= 0.99

    def test_pipeline_with_subset_of_classifiers(self):
        """Test pipeline with specific classifiers only."""
        pipeline = SafetyPipeline(classifiers=["pii"])
        
        report = pipeline.analyze("Test text")
        
        assert len(report.results) == 1
        assert report.results[0].classifier == "pii"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def pipeline(self):
        return SafetyPipeline(classifiers=["pii"], device="cpu")

    def test_very_long_text(self, pipeline):
        """Test handling of long text."""
        long_text = "Hello world. " * 1000  # ~13000 characters
        
        report = pipeline.analyze(long_text)
        
        assert isinstance(report, SafetyReport)
        assert report.processing_time_ms > 0

    def test_unicode_text(self, pipeline):
        """Test handling of unicode characters."""
        unicode_text = "Hello 世界! Привет! مرحبا! 🌍🎉"
        
        report = pipeline.analyze(unicode_text)
        
        assert isinstance(report, SafetyReport)
        assert report.flagged is False

    def test_special_characters(self, pipeline):
        """Test handling of special characters."""
        special_text = "Hello <script>alert('xss')</script> & \"quotes\" 'apostrophes'"
        
        report = pipeline.analyze(special_text)
        
        assert isinstance(report, SafetyReport)

    def test_newlines_and_tabs(self, pipeline):
        """Test handling of newlines and tabs."""
        text_with_whitespace = "Line 1\nLine 2\tTabbed\r\nWindows line"
        
        report = pipeline.analyze(text_with_whitespace)
        
        assert isinstance(report, SafetyReport)

    def test_empty_batch(self, pipeline):
        """Test batch with empty list."""
        reports = pipeline.analyze_batch([])
        
        assert reports == []

    def test_single_item_batch(self, pipeline):
        """Test batch with single item."""
        reports = pipeline.analyze_batch(["Hello"])
        
        assert len(reports) == 1
        assert isinstance(reports[0], SafetyReport)


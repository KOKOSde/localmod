"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from localmod.api.app import create_app, reset_pipeline


@pytest.fixture
def client():
    """Create a test client for the API."""
    reset_pipeline()  # Reset before each test
    app = create_app()
    with TestClient(app) as client:
        yield client
    reset_pipeline()  # Cleanup after


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "LocalMod"
        assert "version" in data
        assert data["docs"] == "/docs"


class TestHealthEndpoint:
    """Tests for the health endpoint."""

    def test_health_check(self, client):
        """Test health check returns status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "models_loaded" in data
        assert "device" in data


class TestClassifiersEndpoint:
    """Tests for the classifiers listing endpoint."""

    def test_list_classifiers(self, client):
        """Test listing available classifiers."""
        response = client.get("/classifiers")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have all classifiers
        assert "toxicity" in data
        assert "pii" in data
        assert "prompt_injection" in data
        assert "spam" in data
        assert "nsfw" in data
        
        # Each should have name, version, description
        for name, info in data.items():
            assert "name" in info
            assert "version" in info
            assert "description" in info


class TestAnalyzeEndpoint:
    """Tests for the analyze endpoint."""

    def test_analyze_safe_text(self, client):
        """Test analyzing safe text with PII classifier only."""
        response = client.post(
            "/analyze",
            json={
                "text": "Hello, how are you today?",
                "classifiers": ["pii"]  # Use only PII to avoid ML model loading
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "flagged" in data
        assert "results" in data
        assert "summary" in data
        assert "processing_time_ms" in data

    def test_analyze_text_with_pii(self, client):
        """Test analyzing text with PII."""
        response = client.post(
            "/analyze",
            json={
                "text": "My email is john@example.com",
                "classifiers": ["pii"]  # Use only PII to avoid ML model loading
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["flagged"] is True
        
        # Find PII result
        pii_result = next((r for r in data["results"] if r["classifier"] == "pii"), None)
        assert pii_result is not None
        assert pii_result["flagged"] is True

    def test_analyze_specific_classifiers(self, client):
        """Test analyzing with specific classifiers."""
        response = client.post(
            "/analyze",
            json={
                "text": "My email is john@example.com",
                "classifiers": ["pii"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should only have PII result
        assert len(data["results"]) == 1
        assert data["results"][0]["classifier"] == "pii"

    def test_analyze_with_explanation(self, client):
        """Test analyzing with explanation."""
        response = client.post(
            "/analyze",
            json={
                "text": "Hello world",
                "classifiers": ["pii"],  # Use only PII to avoid ML model loading
                "include_explanation": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Results should have explanations
        for result in data["results"]:
            assert "explanation" in result

    def test_analyze_empty_text(self, client):
        """Test analyzing empty text returns error."""
        response = client.post(
            "/analyze",
            json={"text": ""}
        )
        
        # Should return validation error
        assert response.status_code == 422

    def test_analyze_whitespace_text(self, client):
        """Test analyzing whitespace-only text returns error."""
        response = client.post(
            "/analyze",
            json={"text": "   "}
        )
        
        # Should return validation error
        assert response.status_code == 422


class TestBatchAnalyzeEndpoint:
    """Tests for the batch analyze endpoint."""

    def test_batch_analyze(self, client):
        """Test batch analysis."""
        response = client.post(
            "/analyze/batch",
            json={
                "texts": [
                    "Hello world",
                    "My email is test@example.com",
                    "Nice weather"
                ],
                "classifiers": ["pii"]  # Use only PII to avoid ML model loading
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["results"]) == 3
        assert "total_flagged" in data
        assert "processing_time_ms" in data

    def test_batch_analyze_specific_classifiers(self, client):
        """Test batch analysis with specific classifiers."""
        response = client.post(
            "/analyze/batch",
            json={
                "texts": ["Hello", "World"],
                "classifiers": ["pii"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        for result in data["results"]:
            assert len(result["results"]) == 1


class TestRedactEndpoint:
    """Tests for the redact endpoint."""

    def test_redact_pii(self, client):
        """Test PII redaction."""
        response = client.post(
            "/redact",
            json={"text": "My email is john@example.com and phone is 555-123-4567"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "original_length" in data
        assert "redacted_text" in data
        assert "redactions" in data
        assert "processing_time_ms" in data
        
        # Check redaction worked
        assert "john@example.com" not in data["redacted_text"]
        # Default replacement is [REDACTED] (from schema) or type-specific label
        assert len(data["redactions"]) >= 1

    def test_redact_custom_replacement(self, client):
        """Test redaction with custom replacement."""
        response = client.post(
            "/redact",
            json={
                "text": "Email: test@example.com",
                "replacement": "***REDACTED***"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "***REDACTED***" in data["redacted_text"]

    def test_redact_no_pii(self, client):
        """Test redaction when no PII present."""
        response = client.post(
            "/redact",
            json={"text": "Hello, how are you today?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["redacted_text"] == "Hello, how are you today?"
        assert len(data["redactions"]) == 0


"""Tests for PII detector classifier."""

import pytest
from localmod.classifiers.pii import PIIDetector, PIIMatch
from localmod.models.base import Severity


class TestPIIDetector:
    """Tests for PIIDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a PII detector instance."""
        detector = PIIDetector()
        detector.load()
        return detector

    def test_detect_email(self, detector):
        """Test email detection."""
        text = "Contact me at john.doe@example.com for details."
        result = detector.predict(text)
        
        assert result.flagged is True
        assert result.confidence == 1.0
        assert "email" in result.categories
        assert result.metadata["total_count"] == 1

    def test_detect_phone_us(self, detector):
        """Test US phone number detection."""
        text = "Call me at 555-123-4567 anytime."
        result = detector.predict(text)
        
        assert result.flagged is True
        assert "phone_us" in result.categories

    def test_detect_ssn(self, detector):
        """Test SSN detection with critical severity."""
        text = "My SSN is 123-45-6789"
        result = detector.predict(text)
        
        assert result.flagged is True
        assert "ssn" in result.categories
        assert result.severity == Severity.CRITICAL

    def test_detect_credit_card(self, detector):
        """Test credit card detection with Luhn validation."""
        # Valid credit card number (passes Luhn check)
        text = "Card number: 4111-1111-1111-1111"
        result = detector.predict(text)
        
        assert result.flagged is True
        assert "credit_card" in result.categories
        assert result.severity == Severity.CRITICAL

    def test_invalid_credit_card(self, detector):
        """Test that invalid credit card numbers are not flagged."""
        # Invalid credit card number (fails Luhn check)
        text = "Not a card: 1234-5678-9012-3456"
        result = detector.predict(text)
        
        # Should not detect as credit card due to Luhn check failure
        if result.flagged:
            assert "credit_card" not in result.categories

    def test_detect_ip_address(self, detector):
        """Test IP address detection."""
        text = "Server IP is 192.168.1.100"
        result = detector.predict(text)
        
        assert result.flagged is True
        assert "ip_address" in result.categories

    def test_invalid_ip_address(self, detector):
        """Test that invalid IP addresses are not flagged."""
        text = "Not an IP: 999.999.999.999"
        result = detector.predict(text)
        
        # Should not flag invalid IP
        if result.flagged:
            assert "ip_address" not in result.categories

    def test_multiple_pii(self, detector):
        """Test detection of multiple PII types."""
        text = "Email: test@example.com, Phone: 555-123-4567, SSN: 123-45-6789"
        result = detector.predict(text)
        
        assert result.flagged is True
        assert len(result.categories) >= 3
        assert result.severity == Severity.CRITICAL  # Due to SSN

    def test_no_pii(self, detector):
        """Test that clean text is not flagged."""
        text = "Hello, this is a normal message with no personal information."
        result = detector.predict(text)
        
        assert result.flagged is False
        assert result.confidence == 0.0
        assert result.severity == Severity.NONE
        assert result.categories == []

    def test_empty_text(self, detector):
        """Test handling of empty text."""
        result = detector.predict("")
        
        assert result.flagged is False
        assert result.confidence == 0.0
        assert result.severity == Severity.NONE

    def test_whitespace_text(self, detector):
        """Test handling of whitespace-only text."""
        result = detector.predict("   \n\t   ")
        
        assert result.flagged is False

    def test_redact_pii(self, detector):
        """Test PII redaction."""
        text = "My email is john@example.com and my phone is 555-123-4567"
        redacted, detections = detector.redact(text)
        
        assert "john@example.com" not in redacted
        assert "[EMAIL]" in redacted
        assert "[PHONE]" in redacted
        assert len(detections) == 2

    def test_redact_with_custom_replacement(self, detector):
        """Test redaction with custom replacement string."""
        text = "Email: john@example.com"
        redacted, detections = detector.redact(text, replacement="***")
        
        assert "john@example.com" not in redacted
        assert "***" in redacted

    def test_redact_no_pii(self, detector):
        """Test redaction when no PII is present."""
        text = "Hello, world!"
        redacted, detections = detector.redact(text)
        
        assert redacted == text
        assert detections == []


class TestPIIMatch:
    """Tests for PIIMatch dataclass."""

    def test_pii_match_creation(self):
        """Test creating a PIIMatch."""
        match = PIIMatch(
            type="email",
            value="test@example.com",
            start=0,
            end=16,
            redacted="[EMAIL]",
        )
        
        assert match.type == "email"
        assert match.value == "test@example.com"
        assert match.start == 0
        assert match.end == 16
        assert match.redacted == "[EMAIL]"


class TestPIIValidation:
    """Tests for PII validation methods."""

    def test_luhn_check_valid(self):
        """Test Luhn algorithm with valid card numbers."""
        # Known valid test card numbers
        assert PIIDetector._luhn_check("4111111111111111") is True
        assert PIIDetector._luhn_check("4111-1111-1111-1111") is True
        assert PIIDetector._luhn_check("5500 0000 0000 0004") is True

    def test_luhn_check_invalid(self):
        """Test Luhn algorithm with invalid card numbers."""
        assert PIIDetector._luhn_check("1234567890123456") is False
        assert PIIDetector._luhn_check("1111111111111112") is False

    def test_ssn_validation_valid(self):
        """Test SSN validation with valid SSNs."""
        assert PIIDetector._validate_ssn("123-45-6789") is True
        assert PIIDetector._validate_ssn("123456789") is True

    def test_ssn_validation_invalid(self):
        """Test SSN validation with invalid SSNs."""
        assert PIIDetector._validate_ssn("900-00-0000") is False  # Starts with 9
        assert PIIDetector._validate_ssn("666-00-0000") is False  # Starts with 666
        assert PIIDetector._validate_ssn("000-00-0000") is False  # Starts with 000

    def test_ip_validation_valid(self):
        """Test IP validation with valid IPs."""
        assert PIIDetector._validate_ip("192.168.1.1") is True
        assert PIIDetector._validate_ip("0.0.0.0") is True
        assert PIIDetector._validate_ip("255.255.255.255") is True

    def test_ip_validation_invalid(self):
        """Test IP validation with invalid IPs."""
        assert PIIDetector._validate_ip("256.0.0.0") is False
        assert PIIDetector._validate_ip("192.168.1.999") is False


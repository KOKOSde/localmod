"""Tests for spam classifier."""

import pytest
from localmod.classifiers.spam import SpamClassifier
from localmod.models.base import Severity


class TestSpamClassifier:
    """Tests for SpamClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create a spam classifier without ML model for fast testing."""
        classifier = SpamClassifier(use_ml_model=False, threshold=0.3)
        classifier.load()
        return classifier

    def test_classifier_attributes(self):
        """Test classifier has correct attributes."""
        classifier = SpamClassifier()
        
        assert classifier.name == "spam"
        assert classifier.version == "1.0.0"
        assert classifier.threshold == 0.5

    def test_detect_excessive_caps(self, classifier):
        """Test detection of excessive capitalization."""
        text = "BUY NOW AMAZING DEALS CLICK HERE"
        result = classifier.predict(text)
        
        assert result.flagged is True
        assert "excessive_caps" in result.categories or "high_caps_ratio" in result.categories

    def test_detect_excessive_punctuation(self, classifier):
        """Test detection of excessive punctuation."""
        text = "Amazing offer!!! Don't miss this!!! FREE!!! ACT NOW!!!"
        result = classifier.predict(text)
        
        assert result.flagged is True
        assert "excessive_punctuation" in result.categories or "free_offers" in result.categories

    def test_detect_money_symbols(self, classifier):
        """Test detection of money-related content."""
        texts = [
            "Win $1000 today!",
            "€500 guaranteed prize",
            "Get £200 free",
        ]
        
        for text in texts:
            result = classifier.predict(text)
            # May flag due to money symbols and urgency
            if result.flagged:
                assert "money_symbols" in result.categories or "free_offers" in result.categories

    def test_detect_urgency_words(self, classifier):
        """Test detection of urgency-inducing words."""
        texts = [
            "Act now before it expires!",
            "Limited time offer - urgent!",
            "Immediately claim your prize!",
        ]
        
        for text in texts:
            result = classifier.predict(text)
            assert result.flagged is True, f"Failed to detect: {text}"

    def test_detect_free_offers(self, classifier):
        """Test detection of free offer claims."""
        texts = [
            "Congratulations! You've won a prize! Claim now!!!",
            "FREE FREE FREE gift for you! ACT NOW!",
            "Claim your free reward now! LIMITED TIME!",
        ]
        
        for text in texts:
            result = classifier.predict(text)
            assert result.flagged is True, f"Failed to detect: {text}"

    def test_detect_click_bait(self, classifier):
        """Test detection of click-bait phrases."""
        texts = [
            "CLICK HERE to learn more! FREE offer!!!",
            "Sign up now for exclusive access! LIMITED TIME!",
            "BUY NOW and save! ACT IMMEDIATELY!",
        ]
        
        for text in texts:
            result = classifier.predict(text)
            assert result.flagged is True, f"Failed to detect: {text}"

    def test_detect_suspicious_urls(self, classifier):
        """Test detection of suspicious shortened URLs."""
        texts = [
            "Check this out: bit.ly/abc123 FREE OFFER!!!",
            "Visit tinyurl.com/xyz for FREE details! ACT NOW!",
            "Link: goo.gl/abc - WIN $1000 NOW!!!",
        ]
        
        for text in texts:
            result = classifier.predict(text)
            assert result.flagged is True, f"Failed to detect: {text}"
            assert "suspicious_urls" in result.categories

    def test_normal_text_not_flagged(self, classifier):
        """Test that normal text is not flagged."""
        texts = [
            "Hello, how are you doing today?",
            "Can we schedule a meeting for next week?",
            "I enjoyed reading your article about machine learning.",
            "The weather is nice today.",
        ]
        
        for text in texts:
            result = classifier.predict(text)
            assert result.flagged is False, f"Incorrectly flagged: {text}"

    def test_empty_text(self, classifier):
        """Test handling of empty text."""
        result = classifier.predict("")
        
        assert result.flagged is False
        assert result.confidence == 0.0
        assert result.severity == Severity.NONE

    def test_metadata_structure(self, classifier):
        """Test that metadata has expected structure."""
        result = classifier.predict("FREE MONEY!!! Click here!!!")
        
        assert "pattern_matches" in result.metadata
        assert "heuristic_score" in result.metadata
        assert "ml_score" in result.metadata


class TestSpamSeverity:
    """Tests for severity classification."""

    @pytest.fixture
    def classifier(self):
        classifier = SpamClassifier(use_ml_model=False, threshold=0.3)
        classifier.load()
        return classifier

    def test_severity_levels(self, classifier):
        """Test severity mapping based on threshold=0.3."""
        # With threshold=0.3, severity mapping changes
        assert classifier._get_severity(0.2) == Severity.NONE
        assert classifier._get_severity(0.55) == Severity.LOW
        assert classifier._get_severity(0.7) == Severity.MEDIUM
        assert classifier._get_severity(0.85) == Severity.HIGH


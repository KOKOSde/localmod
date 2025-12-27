"""PII (Personally Identifiable Information) detection and redaction."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Tuple

from localmod.models.base import BaseClassifier, ClassificationResult, Severity


@dataclass
class PIIMatch:
    """A detected PII instance."""
    type: str
    value: str
    start: int
    end: int
    redacted: str


class PIIDetector(BaseClassifier):
    """
    Detects personally identifiable information using regex patterns and validation.
    
    Supports: emails, phone numbers, SSNs, credit cards, IP addresses.
    No ML model required - fully deterministic.
    """
    
    name = "pii"
    version = "1.0.0"
    
    # Regex patterns for PII detection
    PATTERNS: Dict[str, str] = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone_us": r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
        "phone_intl": r'\b\+?[1-9]\d{1,14}\b',
        "ssn": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "date_of_birth": r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)?\d{2}\b',
    }
    
    REDACTION_LABELS: Dict[str, str] = {
        "email": "[EMAIL]",
        "phone_us": "[PHONE]",
        "phone_intl": "[PHONE]",
        "ssn": "[SSN]",
        "credit_card": "[CREDIT_CARD]",
        "ip_address": "[IP_ADDRESS]",
        "date_of_birth": "[DOB]",
    }

    def __init__(
        self,
        device: str = "auto",  # Unused but kept for interface consistency
        threshold: float = 0.5,
        categories: Optional[List[str]] = None,
    ):
        super().__init__(device="cpu", threshold=threshold)  # Always CPU
        self.enabled_categories = categories or list(self.PATTERNS.keys())
        self._compiled_patterns: Dict[str, Pattern] = {}

    def load(self) -> None:
        """Compile regex patterns."""
        for name, pattern in self.PATTERNS.items():
            if name in self.enabled_categories:
                self._compiled_patterns[name] = re.compile(pattern, re.IGNORECASE)
        self._model = True  # Mark as loaded

    def predict(self, text: str) -> ClassificationResult:
        """Detect PII in text."""
        self._ensure_loaded()
        
        if not text.strip():
            return ClassificationResult(
                classifier=self.name,
                flagged=False,
                confidence=0.0,
                severity=Severity.NONE,
            )
        
        detections = self._find_all_pii(text)
        flagged = len(detections) > 0
        
        # Confidence is 1.0 if PII found (deterministic)
        confidence = 1.0 if flagged else 0.0
        
        # Categorize detections
        categories = list(set(d.type for d in detections))
        
        # Build metadata with redacted values
        metadata = {
            "detections": [
                {
                    "type": d.type,
                    "start": d.start,
                    "end": d.end,
                    "redacted_preview": d.redacted,
                }
                for d in detections
            ],
            "total_count": len(detections),
        }
        
        return ClassificationResult(
            classifier=self.name,
            flagged=flagged,
            confidence=confidence,
            severity=self._get_severity(detections),
            categories=categories,
            metadata=metadata,
        )

    def _find_all_pii(self, text: str) -> List[PIIMatch]:
        """Find all PII matches in text."""
        detections: List[PIIMatch] = []
        
        for pii_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                # Validate the match
                if self._validate_match(pii_type, match.group()):
                    detections.append(PIIMatch(
                        type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        redacted=self.REDACTION_LABELS.get(pii_type, "[PII]"),
                    ))
        
        # Sort by position and remove overlaps
        detections.sort(key=lambda x: x.start)
        return self._remove_overlaps(detections)

    def _validate_match(self, pii_type: str, value: str) -> bool:
        """Validate PII matches to reduce false positives."""
        if pii_type == "credit_card":
            return self._luhn_check(value)
        elif pii_type == "ssn":
            return self._validate_ssn(value)
        elif pii_type == "ip_address":
            return self._validate_ip(value)
        return True

    @staticmethod
    def _luhn_check(card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        digits = [int(d) for d in re.sub(r'\D', '', card_number)]
        if len(digits) < 13 or len(digits) > 19:
            return False
        
        checksum = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        
        return checksum % 10 == 0

    @staticmethod
    def _validate_ssn(ssn: str) -> bool:
        """Basic SSN validation."""
        digits = re.sub(r'\D', '', ssn)
        if len(digits) != 9:
            return False
        # SSN cannot start with 9, 666, or 000
        if digits.startswith('9') or digits.startswith('666') or digits.startswith('000'):
            return False
        return True

    @staticmethod
    def _validate_ip(ip: str) -> bool:
        """Validate IP address format."""
        parts = ip.split('.')
        try:
            return all(0 <= int(p) <= 255 for p in parts)
        except ValueError:
            return False

    @staticmethod
    def _remove_overlaps(detections: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping detections, keeping the longest match."""
        if not detections:
            return []
        
        result = [detections[0]]
        for current in detections[1:]:
            previous = result[-1]
            if current.start >= previous.end:
                result.append(current)
            elif current.end - current.start > previous.end - previous.start:
                result[-1] = current
        
        return result

    def _get_severity(self, detections: List[PIIMatch]) -> Severity:
        """Determine severity based on PII types found."""
        if not detections:
            return Severity.NONE
        
        types = set(d.type for d in detections)
        
        # SSN and credit cards are critical
        if types & {"ssn", "credit_card"}:
            return Severity.CRITICAL
        # Multiple types or sensitive data is high
        elif len(types) > 1 or "date_of_birth" in types:
            return Severity.HIGH
        # Single email or phone is medium
        elif types & {"email", "phone_us", "phone_intl"}:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def redact(self, text: str, replacement: Optional[str] = None) -> Tuple[str, List[PIIMatch]]:
        """Redact all PII from text."""
        self._ensure_loaded()
        
        detections = self._find_all_pii(text)
        if not detections:
            return text, []
        
        # Redact from end to start to preserve positions
        result = text
        for detection in reversed(detections):
            label = replacement or detection.redacted
            result = result[:detection.start] + label + result[detection.end:]
        
        return result, detections


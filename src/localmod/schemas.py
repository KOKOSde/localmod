"""Pydantic models for API request/response validation."""

from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field, validator


class ClassifierType(str, Enum):
    """Available classifier types."""
    TOXICITY = "toxicity"
    PII = "pii"
    PROMPT_INJECTION = "prompt_injection"
    SPAM = "spam"
    NSFW = "nsfw"
    ALL = "all"


class AnalyzeRequest(BaseModel):
    """Request model for content analysis."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    classifiers: List[ClassifierType] = Field(
        default=[ClassifierType.ALL],
        description="Which classifiers to run"
    )
    include_explanation: bool = Field(
        default=False,
        description="Include human-readable explanations"
    )

    @validator("text")
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v


class BatchAnalyzeRequest(BaseModel):
    """Request model for batch content analysis."""
    texts: List[str] = Field(..., min_length=1, max_length=32, description="Texts to analyze")
    classifiers: List[ClassifierType] = Field(default=[ClassifierType.ALL])
    include_explanation: bool = Field(default=False)

    @validator("texts")
    def texts_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Texts list cannot be empty")
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")
        return v


class ClassifierResult(BaseModel):
    """Result from a single classifier."""
    classifier: str
    flagged: bool
    confidence: float = Field(ge=0.0, le=1.0)
    severity: str
    categories: List[str] = []
    metadata: Dict[str, Any] = {}
    explanation: str = ""


class AnalyzeResponse(BaseModel):
    """Response model for content analysis."""
    flagged: bool = Field(description="True if ANY classifier flagged the content")
    results: List[ClassifierResult]
    summary: str = Field(description="Human-readable summary")
    processing_time_ms: float


class BatchAnalyzeResponse(BaseModel):
    """Response model for batch content analysis."""
    results: List[AnalyzeResponse]
    total_flagged: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: List[str]
    device: str


class RedactRequest(BaseModel):
    """Request for PII redaction."""
    text: str = Field(..., min_length=1, max_length=10000)
    replacement: str = Field(default="[REDACTED]", max_length=50)


class RedactResponse(BaseModel):
    """Response for PII redaction."""
    original_length: int
    redacted_text: str
    redactions: List[Dict[str, Any]]
    processing_time_ms: float

"""API route definitions."""

import time
from typing import List

from fastapi import APIRouter, HTTPException, Depends

from localmod import __version__
from localmod.config import get_settings, Settings
from localmod.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    ClassifierResult,
    ClassifierType,
    HealthResponse,
    RedactRequest,
    RedactResponse,
)
from localmod.api.app import get_pipeline
from localmod.pipeline import SafetyPipeline

router = APIRouter()


def get_settings_dep() -> Settings:
    return get_settings()


def get_pipeline_dep() -> SafetyPipeline:
    return get_pipeline()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(pipeline: SafetyPipeline = Depends(get_pipeline_dep)):
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        models_loaded=pipeline.loaded_classifiers,
        device=pipeline.device,
    )


@router.get("/classifiers", tags=["System"])
async def list_classifiers():
    """List available classifiers with descriptions."""
    from localmod.classifiers import CLASSIFIER_REGISTRY
    
    return {
        name: {
            "name": cls.name,
            "version": cls.version,
            "description": cls.__doc__.strip().split('\n')[0] if cls.__doc__ else "",
        }
        for name, cls in CLASSIFIER_REGISTRY.items()
    }


@router.post("/analyze", response_model=AnalyzeResponse, tags=["Moderation"])
async def analyze_content(
    request: AnalyzeRequest,
    pipeline: SafetyPipeline = Depends(get_pipeline_dep),
    settings: Settings = Depends(get_settings_dep),
):
    """
    Analyze text content for safety issues.
    
    Runs specified classifiers (or all if not specified) and returns
    aggregated results with confidence scores and severity levels.
    """
    # Validate text length
    if len(request.text) > settings.max_text_length:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds maximum length of {settings.max_text_length} characters",
        )
    
    # Determine classifiers to run
    classifiers = None
    if ClassifierType.ALL not in request.classifiers:
        classifiers = [c.value for c in request.classifiers]
    
    # Run analysis
    report = pipeline.analyze(
        text=request.text,
        classifiers=classifiers,
        include_explanation=request.include_explanation,
    )
    
    return AnalyzeResponse(
        flagged=report.flagged,
        results=[
            ClassifierResult(**r.to_dict())
            for r in report.results
        ],
        summary=report.summary,
        processing_time_ms=report.processing_time_ms,
    )


@router.post("/analyze/batch", response_model=BatchAnalyzeResponse, tags=["Moderation"])
async def analyze_batch(
    request: BatchAnalyzeRequest,
    pipeline: SafetyPipeline = Depends(get_pipeline_dep),
    settings: Settings = Depends(get_settings_dep),
):
    """
    Analyze multiple texts in a single request.
    
    Maximum batch size is configurable (default: 32).
    """
    if len(request.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum of {settings.max_batch_size}",
        )
    
    # Validate all texts
    for i, text in enumerate(request.texts):
        if len(text) > settings.max_text_length:
            raise HTTPException(
                status_code=400,
                detail=f"Text at index {i} exceeds maximum length",
            )
    
    start_time = time.perf_counter()
    
    # Determine classifiers
    classifiers = None
    if ClassifierType.ALL not in request.classifiers:
        classifiers = [c.value for c in request.classifiers]
    
    # Run batch analysis
    reports = pipeline.analyze_batch(
        texts=request.texts,
        classifiers=classifiers,
        include_explanation=request.include_explanation,
    )
    
    total_time = (time.perf_counter() - start_time) * 1000
    
    return BatchAnalyzeResponse(
        results=[
            AnalyzeResponse(
                flagged=r.flagged,
                results=[ClassifierResult(**cr.to_dict()) for cr in r.results],
                summary=r.summary,
                processing_time_ms=r.processing_time_ms,
            )
            for r in reports
        ],
        total_flagged=sum(1 for r in reports if r.flagged),
        processing_time_ms=total_time,
    )


@router.post("/redact", response_model=RedactResponse, tags=["Moderation"])
async def redact_pii(
    request: RedactRequest,
    pipeline: SafetyPipeline = Depends(get_pipeline_dep),
):
    """
    Redact PII from text.
    
    Returns the redacted text along with details about what was redacted.
    """
    start_time = time.perf_counter()
    
    pii_classifier = pipeline.get_classifier("pii")
    if pii_classifier is None:
        raise HTTPException(status_code=500, detail="PII classifier not available")
    
    # Import here to avoid circular imports
    from localmod.classifiers.pii import PIIDetector
    if not isinstance(pii_classifier, PIIDetector):
        raise HTTPException(status_code=500, detail="Invalid PII classifier")
    
    redacted_text, detections = pii_classifier.redact(request.text, request.replacement)
    
    processing_time = (time.perf_counter() - start_time) * 1000
    
    return RedactResponse(
        original_length=len(request.text),
        redacted_text=redacted_text,
        redactions=[
            {
                "type": d.type,
                "start": d.start,
                "end": d.end,
                "replacement": d.redacted,
            }
            for d in detections
        ],
        processing_time_ms=processing_time,
    )


@router.get("/", tags=["System"])
async def root():
    """API root endpoint."""
    return {
        "name": "LocalMod",
        "version": __version__,
        "description": "Fully offline content moderation API",
        "docs": "/docs",
    }


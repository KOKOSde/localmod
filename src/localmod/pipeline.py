"""Multi-classifier pipeline orchestrator."""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from localmod.classifiers import (
    CLASSIFIER_REGISTRY,
    BaseClassifier,
    ClassificationResult,
    Severity,
)
from localmod.config import get_settings


@dataclass
class SafetyReport:
    """Aggregated report from all classifiers."""
    flagged: bool
    severity: Severity
    results: List[ClassificationResult]
    summary: str
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flagged": self.flagged,
            "severity": self.severity.value,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "metadata": self.metadata,
        }


class SafetyPipeline:
    """
    Orchestrates multiple classifiers for comprehensive content moderation.
    
    Features:
    - Lazy loading of models
    - Parallel execution (optional)
    - Configurable classifiers
    - Aggregated reporting
    """
    
    def __init__(
        self,
        classifiers: Optional[List[str]] = None,
        device: str = "auto",
        parallel: bool = False,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        settings = get_settings()
        
        self.classifier_names = classifiers or list(CLASSIFIER_REGISTRY.keys())
        self.device = device
        self.parallel = parallel
        self.thresholds = thresholds or {}
        
        # Initialize classifiers
        self._classifiers: Dict[str, BaseClassifier] = {}
        for name in self.classifier_names:
            if name in CLASSIFIER_REGISTRY:
                threshold = self.thresholds.get(name, getattr(settings, f"{name}_threshold", 0.5))
                self._classifiers[name] = CLASSIFIER_REGISTRY[name](
                    device=device,
                    threshold=threshold,
                )
        
        self._executor: Optional[ThreadPoolExecutor] = None
        if parallel:
            self._executor = ThreadPoolExecutor(max_workers=len(self._classifiers))

    def load_all(self) -> None:
        """Pre-load all classifier models."""
        for classifier in self._classifiers.values():
            classifier.load()

    def analyze(
        self,
        text: str,
        classifiers: Optional[List[str]] = None,
        include_explanation: bool = False,
    ) -> SafetyReport:
        """
        Analyze text with specified classifiers.
        
        Args:
            text: Text to analyze
            classifiers: Specific classifiers to run (None = all)
            include_explanation: Include human-readable explanations
            
        Returns:
            SafetyReport with aggregated results
        """
        start_time = time.perf_counter()
        
        # Determine which classifiers to run
        active_classifiers = classifiers or list(self._classifiers.keys())
        active_classifiers = [c for c in active_classifiers if c in self._classifiers]
        
        # Run classifiers
        results: List[ClassificationResult] = []
        
        if self.parallel and self._executor is not None:
            # Parallel execution
            futures = {
                name: self._executor.submit(self._classifiers[name].predict, text)
                for name in active_classifiers
            }
            for name, future in futures.items():
                result = future.result()
                if include_explanation:
                    result.explanation = self._classifiers[name].get_explanation(text, result)
                results.append(result)
        else:
            # Sequential execution
            for name in active_classifiers:
                result = self._classifiers[name].predict(text)
                if include_explanation:
                    result.explanation = self._classifiers[name].get_explanation(text, result)
                results.append(result)
        
        # Aggregate results
        flagged = any(r.flagged for r in results)
        max_severity = max((r.severity for r in results), key=lambda s: list(Severity).index(s))
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return SafetyReport(
            flagged=flagged,
            severity=max_severity,
            results=results,
            summary=self._generate_summary(results),
            processing_time_ms=processing_time,
            metadata={
                "classifiers_run": active_classifiers,
                "device": self.device,
            },
        )

    def analyze_batch(
        self,
        texts: List[str],
        classifiers: Optional[List[str]] = None,
        include_explanation: bool = False,
    ) -> List[SafetyReport]:
        """Analyze multiple texts."""
        return [
            self.analyze(text, classifiers, include_explanation)
            for text in texts
        ]

    def _generate_summary(self, results: List[ClassificationResult]) -> str:
        """Generate human-readable summary of results."""
        flagged_results = [r for r in results if r.flagged]
        
        if not flagged_results:
            return "Content passed all safety checks."
        
        issues = []
        for r in flagged_results:
            issue = f"{r.classifier} ({r.severity.value})"
            if r.categories:
                issue += f": {', '.join(r.categories)}"
            issues.append(issue)
        
        return f"Content flagged for: {'; '.join(issues)}"

    def get_classifier(self, name: str) -> Optional[BaseClassifier]:
        """Get a specific classifier instance."""
        return self._classifiers.get(name)

    @property
    def loaded_classifiers(self) -> List[str]:
        """List of loaded classifier names."""
        return [name for name, c in self._classifiers.items() if c.is_loaded]

    @property
    def available_classifiers(self) -> List[str]:
        """List of available classifier names."""
        return list(self._classifiers.keys())


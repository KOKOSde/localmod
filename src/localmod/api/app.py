"""FastAPI application factory."""

from contextlib import contextmanager
from typing import Generator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from localmod.config import get_settings
from localmod.pipeline import SafetyPipeline


# Global pipeline instance
_pipeline: Optional[SafetyPipeline] = None


def get_pipeline() -> SafetyPipeline:
    """Get the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        settings = get_settings()
        _pipeline = SafetyPipeline(device=settings.device)
        if not settings.lazy_load:
            _pipeline.load_all()
    return _pipeline


def reset_pipeline() -> None:
    """Reset the global pipeline (for testing)."""
    global _pipeline
    _pipeline = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="LocalMod",
        description="Fully offline content moderation API",
        version="0.1.0",
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Import and include routes
    from localmod.api.routes import router
    app.include_router(router)
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize pipeline on startup."""
        settings = get_settings()
        if not settings.lazy_load:
            pipeline = get_pipeline()
            pipeline.load_all()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        reset_pipeline()
    
    return app


# Default app instance
app = create_app()



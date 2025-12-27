.PHONY: install dev test lint format docker-build docker-run serve clean help

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# Installation
# =============================================================================

install: ## Install package in production mode
	pip install -e .

dev: ## Install package with dev dependencies
	pip install -e ".[all]"

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests with coverage
	pytest tests/ -v --cov=localmod --cov-report=term-missing

test-fast: ## Run fast tests only (skip ML model tests)
	pytest tests/ -v -m "not slow" --ignore=tests/test_integration.py

test-unit: ## Run unit tests only
	pytest tests/test_base.py tests/test_classifiers/ -v

test-api: ## Run API tests only
	pytest tests/test_api.py tests/test_pipeline.py -v

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linter
	flake8 src/ tests/
	mypy src/localmod/

format: ## Format code
	black src/ tests/
	isort src/ tests/

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build CPU Docker image
	docker build -f docker/Dockerfile -t localmod:latest .

docker-build-gpu: ## Build GPU Docker image
	docker build -f docker/Dockerfile.gpu -t localmod:gpu .

docker-run: ## Run CPU Docker container
	docker run -p 8000:8000 --name localmod localmod:latest

docker-run-gpu: ## Run GPU Docker container
	docker run --gpus all -p 8000:8000 --name localmod-gpu localmod:gpu

docker-stop: ## Stop and remove containers
	docker stop localmod localmod-gpu 2>/dev/null || true
	docker rm localmod localmod-gpu 2>/dev/null || true

docker-compose-up: ## Start services with docker-compose
	docker-compose -f docker/docker-compose.yml up -d

docker-compose-up-gpu: ## Start GPU services with docker-compose
	docker-compose -f docker/docker-compose.yml --profile gpu up -d

docker-compose-down: ## Stop docker-compose services
	docker-compose -f docker/docker-compose.yml down

docker-compose-logs: ## View docker-compose logs
	docker-compose -f docker/docker-compose.yml logs -f

# =============================================================================
# Development
# =============================================================================

serve: ## Start development server with auto-reload
	python -m localmod.cli serve --reload

serve-prod: ## Start production server
	python -m localmod.cli serve --workers 4

download-models: ## Pre-download all models
	python -m localmod.cli download

list-classifiers: ## List available classifiers
	python -m localmod.cli list

analyze: ## Analyze text (usage: make analyze TEXT="your text here")
	python -m localmod.cli analyze "$(TEXT)"

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	rm -rf .coverage htmlcov/ coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-docker: ## Clean Docker images and containers
	docker stop localmod localmod-gpu 2>/dev/null || true
	docker rm localmod localmod-gpu 2>/dev/null || true
	docker rmi localmod:latest localmod:gpu 2>/dev/null || true
	docker volume rm localmod-model-cache 2>/dev/null || true

clean-all: clean clean-docker ## Clean everything

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "LocalMod - Fully Offline Content Moderation API"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'


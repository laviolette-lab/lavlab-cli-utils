.DEFAULT_GOAL := help

.PHONY: help test cov lint format fix types docs serve-docs build build-mac build-linux clean install pre-commit docker-build docker-wheel

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install hatch and create the dev environments
	pip install hatch
	hatch env create

test: ## Run tests
	hatch run test:test

cov: ## Run tests with coverage report
	hatch run test:cov

lint: ## Run Ruff linter
	hatch run lint:check

format: ## Format code with Ruff
	hatch run lint:format

fix: ## Auto-fix lint issues and format
	hatch run lint:all

types: ## Run mypy type checking
	hatch run types:check

docs: ## Build documentation
	hatch run docs:build-docs

serve-docs: ## Serve documentation locally
	hatch run docs:serve-docs

# Not `hatch build`: the wheel embeds a Nuitka-compiled binary produced by
# setup.py's build_py. Needs the Ice wheel, omero-py and build-requirements.txt
# installed in the current environment (see README).
build: ## Build the Nuitka wheel
	python setup.py bdist_wheel

build-mac: ## Build the macOS arm64 wheel (Python 3.12 required)
	LAVLAB_TARGET=macos-arm64 python setup.py bdist_wheel

build-linux: ## Build the Linux x86_64 wheel (Python 3.12 required)
	LAVLAB_TARGET=linux-x86_64 python setup.py bdist_wheel

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ site/ htmlcov/ src/lavlab/bin/
	rm -f coverage.xml .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true

pre-commit: ## Install and run pre-commit hooks
	pre-commit install
	pre-commit run --all-files

docker-build: ## Build the Nuitka build-environment Docker image
	docker build -t lavlab-builder .

docker-wheel: ## Build the wheel inside the Docker build environment
	docker run --rm -v "$$(pwd)":/src -v "$$(pwd)/dist":/out lavlab-builder

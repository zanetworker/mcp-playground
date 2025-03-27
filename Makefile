.PHONY: clean test install dev lint format build

# Default target
all: clean test build

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run tests
test:
	pytest

# Install the package
install:
	pip install .

# Install in development mode
dev:
	pip install -e .
	pip install -r requirements.txt

# Run linting
lint:
	pylint mcp_sse_client tests

# Format code
format:
	black mcp_sse_client tests

# Build package
build: clean
	python setup.py sdist bdist_wheel

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Clean, test, and build the package"
	@echo "  clean    - Remove build artifacts"
	@echo "  test     - Run tests"
	@echo "  install  - Install the package"
	@echo "  dev      - Install in development mode"
	@echo "  lint     - Run linting"
	@echo "  format   - Format code"
	@echo "  build    - Build package"
	@echo "  help     - Show this help message"

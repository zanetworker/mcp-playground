# Development Guide

This document provides information for developers who want to contribute to the MCP SSE Client Python project.

## Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/zanetworker/mcp-playground.git
cd mcp-playground

# Install development dependencies
make dev
```

## Available Make Commands

The project includes a Makefile with common commands:

```bash
make clean    # Remove build artifacts
make test     # Run tests
make install  # Install the package
make dev      # Install in development mode
make lint     # Run linting
make format   # Format code
make build    # Build package
make help     # Show help message
```

## Running Tests

The project includes unit tests to ensure functionality works as expected:

```bash
# Run tests
make test
```

## Contributing

Contributions are welcome! Here are some ways you can contribute to this project:

1. Report bugs and request features by creating issues
2. Submit pull requests to fix bugs or add new features
3. Improve documentation
4. Write tests to increase code coverage

Please follow these steps when contributing:

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Add tests for your changes
4. Make your changes
5. Run the tests to ensure they pass
6. Submit a pull request

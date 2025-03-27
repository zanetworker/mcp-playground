# MCP SSE Client Python

A Python client for interacting with Model Context Protocol (MCP) endpoints using Server-Sent Events (SSE).

## Overview

This client provides a simple interface for:
- Listing available tools from an MCP endpoint
- Invoking tools with parameters
- Handling tool invocation results

## Installation

### From Source

1. Clone the repository:
```bash
git clone https://github.com/zanetworker/mcp-sse-client-python.git
cd mcp-sse-client-python
```

2. Install the package in development mode:
```bash
pip install -e .
```

### Using pip (once published)

```bash
pip install mcp-sse-client
```

## Usage

### Basic Example

```python
import asyncio
from mcp_sse_client import MCPClient

async def main():
    # Initialize the client with your MCP endpoint
    client = MCPClient("http://localhost:8000/sse")
    
    # List available tools
    tools = await client.list_tools()
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
        print("  Parameters:")
        for param in tool.parameters:
            print(f"    - {param.name} ({param.parameter_type}): {param.description}")
    
    # Invoke a tool with parameters
    result = await client.invoke_tool(
        "calculator", 
        {"x": 10, "y": 5, "operation": "add"}
    )
    print(f"\nTool result: {result.content}")
    print(f"Error code: {result.error_code}")

if __name__ == "__main__":
    asyncio.run(main())
```

See the `mcp_sse_client/examples` directory for more examples.

## API Reference

### MCPClient

```python
client = MCPClient(endpoint)
```

- `endpoint`: The MCP endpoint URL (must be http or https)

#### Methods

##### `async list_tools() -> List[ToolDef]`

Lists available tools from the MCP endpoint.

Returns:
- A list of `ToolDef` objects describing available tools

##### `async invoke_tool(tool_name: str, kwargs: Dict[str, Any]) -> ToolInvocationResult`

Invokes a specific tool with parameters.

Parameters:
- `tool_name`: Name of the tool to invoke
- `kwargs`: Dictionary of parameters to pass to the tool

Returns:
- A `ToolInvocationResult` containing the tool's response

### Data Classes

#### ToolDef

Represents a tool definition.

Attributes:
- `name`: Tool name
- `description`: Tool description
- `parameters`: List of `ToolParameter` objects
- `metadata`: Optional dictionary of additional metadata
- `identifier`: Tool identifier (defaults to name)

#### ToolParameter

Represents a parameter for a tool.

Attributes:
- `name`: Parameter name
- `parameter_type`: Parameter type (e.g., "string", "number")
- `description`: Parameter description

#### ToolInvocationResult

Represents the result of a tool invocation.

Attributes:
- `content`: Result content as a string
- `error_code`: Error code (0 for success, 1 for error)

## Requirements

- Python 3.7+
- mcp library
- pydantic

See `requirements.txt` for specific version requirements.

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/zanetworker/mcp-sse-client-python.git
cd mcp-sse-client-python

# Install development dependencies
make dev
```

### Available Make Commands

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

### Running Tests

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

## License

[Add license information here]

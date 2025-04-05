# MCP SSE Client Python

A Python client for interacting with Model Context Protocol (MCP) endpoints using Server-Sent Events (SSE).

## Overview

This client provides a simple interface for:
- Listing available tools from an MCP endpoint
- Invoking tools with parameters
- Handling tool invocation results
- Integrating with LLMs (OpenAI, Anthropic) for AI-driven tool selection

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

### LLM Integration Example

```python
import asyncio
import os
from mcp_sse_client import MCPClient, OpenAIBridge

async def main():
    # Initialize the client with your MCP endpoint
    client = MCPClient("http://localhost:8000/sse")
    
    # Create an OpenAI bridge
    openai_bridge = OpenAIBridge(
        client,
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o"  # or any other OpenAI model
    )
    
    # Process a user query
    result = await openai_bridge.process_query(
        "I need to convert a PDF document to text. Can you help me with that?"
    )
    
    # The LLM will automatically select the appropriate tool and parameters
    if result["tool_call"]:
        print(f"Tool selected: {result['tool_call']['name']}")
        print(f"Parameters: {result['tool_call']['parameters']}")
        print(f"Result: {result['tool_result'].content}")
    else:
        print("No tool was selected by the LLM")

if __name__ == "__main__":
    asyncio.run(main())
```

See the `mcp_sse_client/examples` directory for more programmatic examples.

### Interactive Testing with Streamlit App

This project includes a Streamlit application for interactively testing the MCP client and LLM integrations.

**Features:**
- Connect to any MCP SSE endpoint.
- Select between OpenAI and Anthropic LLM providers.
- View available tools and their parameters.
- Chat interface to send queries to the LLM.
- Visualize LLM reasoning, tool selection, and tool results.
- Conversation history management.
- Smart display and extraction of tool output (JSON, Markdown, Text).

**To run the Streamlit app:**
1. Navigate to the `mcp-streamlit-app` directory:
   ```bash
   cd mcp-streamlit-app
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
   (Alternatively, use the `./run.sh` script in the `mcp-streamlit-app` directory).

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

### LLM Integration

The library provides integration with popular LLM providers to enable AI-driven tool selection and invocation.

#### OpenAIBridge

```python
bridge = OpenAIBridge(mcp_client, api_key, model="gpt-4o")
```

- `mcp_client`: An initialized MCPClient instance
- `api_key`: OpenAI API key
- `model`: OpenAI model to use (default: gpt-4o)

#### AnthropicBridge

```python
bridge = AnthropicBridge(mcp_client, api_key, model="claude-3-opus-20240229")
```

- `mcp_client`: An initialized MCPClient instance
- `api_key`: Anthropic API key
- `model`: Anthropic model to use (default: claude-3-opus-20240229)

#### Common Bridge Methods

##### `async fetch_tools() -> List[ToolDef]`

Fetches available tools from the MCP endpoint.

##### `async process_query(query: str) -> Dict[str, Any]`

Processes a user query through the LLM and executes any tool calls.

Returns a dictionary containing:
- `llm_response`: The raw LLM response
- `tool_call`: The parsed tool call (if any)
- `tool_result`: The result of the tool invocation (if any)

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
- `required`: Whether the parameter is required
- `default`: Default value for the parameter

#### ToolInvocationResult

Represents the result of a tool invocation.

Attributes:
- `content`: Result content as a string
- `error_code`: Error code (0 for success, 1 for error)

## Requirements

- Python 3.7+
- `requests`
- `sseclient-py`
- `pydantic`
- `openai` (for OpenAI integration)
- `anthropic` (for Anthropic integration)
- `streamlit` (for the interactive test app)
- `Markdown` (for the interactive test app)

See `requirements.txt` (root) and `mcp-streamlit-app/requirements.txt` for specific version requirements.

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

This project is licensed under the MIT License - see the LICENSE file for details.

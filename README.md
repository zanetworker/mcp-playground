# MCP Playground

A Python toolkit for interacting with remote Model Context Protocol (MCP) endpoints. Supports **Streamable HTTP** (default) and **SSE** transports, with LLM-powered tool selection across multiple providers.

## Quick Start

```bash
git clone https://github.com/zanetworker/mcp-playground.git
cd mcp-playground
pip install -e ".[all]"
```

## Usage

### Context Manager (recommended)

```python
import asyncio
from mcp_playground import MCPClient

async def main():
    async with MCPClient("http://localhost:8000/mcp") as client:
        tools = await client.list_tools()
        print(f"Found {len(tools)} tools")

        result = await client.call_tool("add", arguments={"a": 10, "b": 5})
        print(f"Result: {result.content}")

asyncio.run(main())
```

### SSE Transport (legacy servers)

```python
async with MCPClient("http://localhost:8000/sse", transport="sse") as client:
    tools = await client.list_tools()
```

### Custom Auth / Headers

```python
import httpx

http = httpx.AsyncClient(headers={"Authorization": "Bearer token"})
async with MCPClient("http://localhost:8000/mcp", http_client=http) as client:
    tools = await client.list_tools()
```

### One-Shot Usage

For simple scripts that don't need a persistent connection:

```python
client = MCPClient("http://localhost:8000/mcp")
tools = await client.list_tools()  # auto-connects and disconnects
```

## LLM-Powered Tool Selection

Let an LLM choose and execute the right MCP tool based on natural language:

```python
import os
from mcp_playground import MCPClient
from mcp_playground.llm_bridge import OpenAIBridge

async with MCPClient("http://localhost:8000/mcp") as client:
    bridge = OpenAIBridge(client, api_key=os.environ["OPENAI_API_KEY"])
    result = await bridge.process_query("Add 10 and 5")

    if result["tool_call"]:
        print(f"Tool: {result['tool_call']['name']}")
        print(f"Result: {result['tool_result'].content}")
```

### Supported LLM Providers

| Provider | Bridge Class | Default Model |
|----------|-------------|---------------|
| OpenAI | `OpenAIBridge` | gpt-4o |
| Anthropic | `AnthropicBridge` | claude-sonnet-4-20250514 |
| Ollama | `OllamaBridge` | llama3.3 |
| OpenRouter | `OpenRouterBridge` | any model ID |

## Interactive Playground (Streamlit)

```bash
cd mcp-streamlit-app
pip install -r requirements.txt
streamlit run app.py
```

Features:
- Transport selector (Streamable HTTP / SSE)
- Multi-provider LLM support (OpenAI, Anthropic, Ollama, OpenRouter)
- Chat modes: Auto, Chat-only, Tools-only
- Tool discovery and invocation
- Real-time connection status

## API Reference

### MCPClient

```python
client = MCPClient(
    endpoint="http://localhost:8000/mcp",
    transport="streamable-http",  # or "sse"
    timeout=30.0,
    http_client=None,  # optional httpx.AsyncClient
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `list_tools()` | List available tools from the MCP endpoint |
| `call_tool(name, arguments={})` | Invoke a tool with parameters |
| `invoke_tool(name, kwargs)` | Backward-compatible alias for `call_tool()` |
| `check_connection()` | Check if the endpoint is reachable |
| `get_endpoint_info()` | Get endpoint metadata |
| `connect()` / `disconnect()` | Manual session lifecycle |

### ToolInvocationResult

```python
result = await client.call_tool("add", arguments={"a": 1, "b": 2})
result.content             # Text content as string
result.error_code          # 0 for success, 1 for error
result.structured_content  # Structured output (dict), if available
result.raw_content         # Raw content blocks from the SDK
```

### Error Handling

```python
from mcp_playground import MCPClient, MCPConnectionError, MCPTimeoutError

try:
    async with MCPClient("http://localhost:8000/mcp") as client:
        tools = await client.list_tools()
except MCPConnectionError as e:
    print(f"Connection failed: {e}")
except MCPTimeoutError as e:
    print(f"Timed out: {e}")
```

## Requirements

- Python >= 3.10
- mcp >= 1.26.0
- httpx >= 0.28.0
- pydantic >= 2.0.0

Optional (for LLM integration):
- openai >= 2.28.0
- anthropic >= 0.84.0
- ollama >= 0.6.0

## Development

```bash
pip install -e ".[dev]"
pytest                # run tests
black .               # format code
ruff check .          # lint
```

## License

MIT

"""
MCP SSE Client - A Python client for interacting with Model Context Protocol (MCP) endpoints.

This package provides a client for connecting to MCP endpoints using Server-Sent Events (SSE),
listing available tools, and invoking tools with parameters. It also includes LLM integration
for AI-driven tool selection and invocation.
"""

from mcp_sse_client.client import MCPClient, ToolDef, ToolParameter, ToolInvocationResult

# Import LLM bridge classes for easier access
try:
    from mcp_sse_client.llm_bridge import LLMBridge, OpenAIBridge, AnthropicBridge
    __all__ = [
        "MCPClient", "ToolDef", "ToolParameter", "ToolInvocationResult",
        "LLMBridge", "OpenAIBridge", "AnthropicBridge"
    ]
except ImportError:
    # LLM dependencies might not be installed
    __all__ = ["MCPClient", "ToolDef", "ToolParameter", "ToolInvocationResult"]

__version__ = "0.1.0"

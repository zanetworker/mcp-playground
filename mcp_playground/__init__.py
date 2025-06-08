"""
MCP Playground - A comprehensive Python toolkit for interacting with remote Model Context Protocol (MCP) endpoints.

This package provides clients for connecting to remote MCP endpoints using various protocols:
- Server-Sent Events (SSE) - Currently supported
- Streamable HTTP - Planned for future releases

Features include tool discovery, invocation, and LLM integration for AI-driven tool selection.
"""

from mcp_playground.client import MCPClient, ToolDef, ToolParameter, ToolInvocationResult

# Import LLM bridge classes for easier access
try:
    from mcp_playground.llm_bridge import LLMBridge, OpenAIBridge, AnthropicBridge
    __all__ = [
        "MCPClient", "ToolDef", "ToolParameter", "ToolInvocationResult",
        "LLMBridge", "OpenAIBridge", "AnthropicBridge"
    ]
except ImportError:
    # LLM dependencies might not be installed
    __all__ = ["MCPClient", "ToolDef", "ToolParameter", "ToolInvocationResult"]

__version__ = "0.2.0"

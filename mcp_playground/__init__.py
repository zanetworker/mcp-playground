"""
MCP Playground - A comprehensive Python toolkit for interacting with remote Model Context Protocol (MCP) endpoints.

Supports Streamable HTTP (default) and SSE transports.
"""

from mcp_playground.client import (
    MCPClient,
    ToolDef,
    ToolParameter,
    ToolInvocationResult,
    MCPConnectionError,
    MCPTimeoutError,
)

try:
    from mcp_playground.llm_bridge import LLMBridge, OpenAIBridge, AnthropicBridge
    __all__ = [
        "MCPClient", "ToolDef", "ToolParameter", "ToolInvocationResult",
        "MCPConnectionError", "MCPTimeoutError",
        "LLMBridge", "OpenAIBridge", "AnthropicBridge",
    ]
except ImportError:
    __all__ = [
        "MCPClient", "ToolDef", "ToolParameter", "ToolInvocationResult",
        "MCPConnectionError", "MCPTimeoutError",
    ]

__version__ = "1.0.0"

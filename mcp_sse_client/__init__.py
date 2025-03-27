"""
MCP SSE Client - A Python client for interacting with Model Context Protocol (MCP) endpoints.

This package provides a client for connecting to MCP endpoints using Server-Sent Events (SSE),
listing available tools, and invoking tools with parameters.
"""

from mcp_sse_client.client import MCPClient, ToolDef, ToolParameter, ToolInvocationResult

__all__ = ["MCPClient", "ToolDef", "ToolParameter", "ToolInvocationResult"]
__version__ = "0.1.0"

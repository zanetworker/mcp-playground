"""
MCP Client - A Python client for interacting with Model Context Protocol (MCP) endpoints.

Supports both Streamable HTTP (default) and SSE transports.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from dataclasses import dataclass, field

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Represents a parameter for a tool."""
    name: str
    parameter_type: str
    description: str
    required: bool = False
    default: Any = None


@dataclass
class ToolDef:
    """Represents a tool definition."""
    name: str
    description: str
    parameters: List[ToolParameter]
    metadata: Optional[Dict[str, Any]] = None
    identifier: str = ""


@dataclass
class ToolInvocationResult:
    """Represents the result of a tool invocation."""
    content: str
    error_code: int
    structured_content: Optional[Dict[str, Any]] = None
    raw_content: Optional[List[Any]] = field(default_factory=list)


class MCPConnectionError(Exception):
    """Exception raised when MCP connection fails."""
    pass


class MCPTimeoutError(Exception):
    """Exception raised when MCP operation times out."""
    pass


class MCPClient:
    """Client for interacting with Model Context Protocol (MCP) endpoints.

    Supports both Streamable HTTP (default) and SSE transports.

    Usage as context manager (recommended):
        async with MCPClient("http://localhost:8000/mcp") as client:
            tools = await client.list_tools()
            result = await client.call_tool("add", arguments={"a": 1, "b": 2})

    Usage with SSE transport:
        async with MCPClient("http://localhost:8000/sse", transport="sse") as client:
            tools = await client.list_tools()

    Usage with custom httpx client:
        import httpx
        http = httpx.AsyncClient(headers={"Authorization": "Bearer token"})
        async with MCPClient("http://localhost:8000/mcp", http_client=http) as client:
            ...

    One-shot usage (opens and closes connection per operation):
        client = MCPClient("http://localhost:8000/mcp")
        tools = await client.list_tools()
    """

    def __init__(
        self,
        endpoint: str,
        transport: str = "streamable-http",
        timeout: float = 30.0,
        http_client: Optional[Any] = None,
    ):
        """Initialize MCP client.

        Args:
            endpoint: The MCP endpoint URL (http or https)
            transport: Transport type - "streamable-http" (default) or "sse"
            timeout: Connection timeout in seconds
            http_client: Optional pre-configured httpx.AsyncClient for custom auth/headers

        Raises:
            ValueError: If endpoint is not a valid HTTP(S) URL or transport is invalid
        """
        parsed = urlparse(endpoint)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Endpoint must be an HTTP(S) URL, got: {endpoint}")

        if transport not in ("streamable-http", "sse"):
            raise ValueError(f"Transport must be 'streamable-http' or 'sse', got: {transport}")

        self.endpoint = endpoint
        self.transport = transport
        self.timeout = timeout
        self.http_client = http_client

        # Session state
        self._session: Optional[ClientSession] = None
        self._transport_context = None
        self._session_context = None
        self._streams = None

    async def __aenter__(self) -> "MCPClient":
        """Open transport connection and initialize session."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close session and transport connection."""
        await self.disconnect()

    async def connect(self) -> None:
        """Establish connection to the MCP server.

        Raises:
            MCPConnectionError: If connection fails
            MCPTimeoutError: If connection times out
        """
        try:
            if self.transport == "streamable-http":
                kwargs = {"url": self.endpoint}
                if self.http_client is not None:
                    kwargs["http_client"] = self.http_client
                self._transport_context = streamable_http_client(**kwargs)
            else:
                self._transport_context = sse_client(self.endpoint)

            streams = await self._transport_context.__aenter__()

            # streamable_http_client returns (read, write, _), sse_client returns (read, write)
            if len(streams) == 3:
                read_stream, write_stream = streams[0], streams[1]
            else:
                read_stream, write_stream = streams[0], streams[1]

            self._session_context = ClientSession(read_stream, write_stream)
            self._session = await self._session_context.__aenter__()

            await asyncio.wait_for(self._session.initialize(), timeout=self.timeout)
            logger.info(f"Connected to {self.endpoint} via {self.transport}")

        except asyncio.TimeoutError:
            await self._cleanup()
            raise MCPTimeoutError(f"Connection to {self.endpoint} timed out after {self.timeout}s")
        except Exception as e:
            await self._cleanup()
            raise MCPConnectionError(f"Failed to connect to {self.endpoint}: {e}") from e

    async def disconnect(self) -> None:
        """Close the connection."""
        await self._cleanup()
        logger.info(f"Disconnected from {self.endpoint}")

    async def _cleanup(self) -> None:
        """Clean up session and transport resources."""
        if self._session_context is not None:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_context = None
            self._session = None

        if self._transport_context is not None:
            try:
                await self._transport_context.__aexit__(None, None, None)
            except Exception:
                pass
            self._transport_context = None

    def _ensure_connected(self) -> ClientSession:
        """Ensure we have an active session, raise if not."""
        if self._session is None:
            raise MCPConnectionError(
                "Not connected. Use 'async with MCPClient(...) as client:' "
                "or call 'await client.connect()' first."
            )
        return self._session

    async def _one_shot(self, operation):
        """Execute a single operation with auto-connect/disconnect.

        Used when the client is not used as a context manager.
        """
        await self.connect()
        try:
            return await operation()
        finally:
            await self.disconnect()

    async def list_tools(self) -> List[ToolDef]:
        """List available tools from the MCP endpoint.

        Returns:
            List of ToolDef objects describing available tools
        """
        if self._session is None:
            return await self._one_shot(self._list_tools)
        return await self._list_tools()

    async def _list_tools(self) -> List[ToolDef]:
        session = self._ensure_connected()
        tools_result = await session.list_tools()
        tools = []

        for tool in tools_result.tools:
            parameters = []
            required_params = tool.inputSchema.get("required", [])
            for param_name, param_schema in tool.inputSchema.get("properties", {}).items():
                parameters.append(
                    ToolParameter(
                        name=param_name,
                        parameter_type=param_schema.get("type", "string"),
                        description=param_schema.get("description", ""),
                        required=param_name in required_params,
                        default=param_schema.get("default"),
                    )
                )
            tools.append(
                ToolDef(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=parameters,
                    metadata={"endpoint": self.endpoint},
                    identifier=tool.name,
                )
            )
        return tools

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> ToolInvocationResult:
        """Invoke a specific tool with parameters.

        Args:
            tool_name: Name of the tool to invoke
            arguments: Dictionary of arguments to pass to the tool

        Returns:
            ToolInvocationResult containing the tool's response
        """
        if self._session is None:
            return await self._one_shot(lambda: self._call_tool(tool_name, arguments))
        return await self._call_tool(tool_name, arguments)

    async def _call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> ToolInvocationResult:
        session = self._ensure_connected()
        result = await session.call_tool(tool_name, arguments=arguments or {})

        # Build text content from content blocks
        text_parts = []
        raw_content = []
        for block in result.content:
            raw_content.append(block)
            if hasattr(block, 'text'):
                text_parts.append(block.text)
            else:
                text_parts.append(block.model_dump_json())

        return ToolInvocationResult(
            content="\n".join(text_parts),
            error_code=1 if result.isError else 0,
            structured_content=getattr(result, 'structuredContent', None),
            raw_content=raw_content,
        )

    # Backward compatibility alias
    async def invoke_tool(self, tool_name: str, kwargs: Dict[str, Any]) -> ToolInvocationResult:
        """Backward-compatible alias for call_tool()."""
        return await self.call_tool(tool_name, arguments=kwargs)

    async def check_connection(self) -> bool:
        """Check if the MCP endpoint is reachable."""
        try:
            await self.list_tools()
            return True
        except Exception:
            return False

    def get_endpoint_info(self) -> Dict[str, Any]:
        """Get information about the configured endpoint."""
        parsed = urlparse(self.endpoint)
        return {
            "endpoint": self.endpoint,
            "transport": self.transport,
            "scheme": parsed.scheme,
            "hostname": parsed.hostname,
            "port": parsed.port,
            "path": parsed.path,
            "timeout": self.timeout,
            "connected": self._session is not None,
        }

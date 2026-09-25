"""
Tests for the MCP Client.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from mcp_playground import MCPClient, ToolDef, ToolParameter, ToolInvocationResult
from mcp_playground.client import MCPConnectionError, MCPTimeoutError


class TestMCPClientInit:
    """Test MCPClient initialization."""

    def test_init_valid_endpoint_default_transport(self):
        client = MCPClient("http://localhost:8000/mcp")
        assert client.endpoint == "http://localhost:8000/mcp"
        assert client.transport == "streamable-http"
        assert client.timeout == 30.0

    def test_init_valid_endpoint_sse_transport(self):
        client = MCPClient("http://localhost:8000/sse", transport="sse")
        assert client.transport == "sse"

    def test_init_invalid_scheme(self):
        with pytest.raises(ValueError, match="HTTP"):
            MCPClient("ftp://example.com/mcp")

    def test_init_invalid_transport(self):
        with pytest.raises(ValueError, match="Transport"):
            MCPClient("http://localhost:8000/mcp", transport="websocket")

    def test_init_custom_timeout(self):
        client = MCPClient("http://localhost:8000/mcp", timeout=60.0)
        assert client.timeout == 60.0

    def test_init_not_connected(self):
        client = MCPClient("http://localhost:8000/mcp")
        assert client._session is None


class TestMCPClientEndpointInfo:
    """Test get_endpoint_info method."""

    def test_endpoint_info_streamable_http(self):
        client = MCPClient("http://localhost:8000/mcp")
        info = client.get_endpoint_info()
        assert info["endpoint"] == "http://localhost:8000/mcp"
        assert info["transport"] == "streamable-http"
        assert info["scheme"] == "http"
        assert info["hostname"] == "localhost"
        assert info["port"] == 8000
        assert info["path"] == "/mcp"
        assert info["connected"] is False

    def test_endpoint_info_sse(self):
        client = MCPClient("https://example.com/sse", transport="sse")
        info = client.get_endpoint_info()
        assert info["transport"] == "sse"
        assert info["scheme"] == "https"


class TestMCPClientContextManager:
    """Test context manager behavior."""

    @pytest.mark.asyncio
    @patch("mcp_playground.client.streamable_http_client")
    @patch("mcp_playground.client.ClientSession")
    async def test_context_manager_connects_and_disconnects(self, mock_session_cls, mock_transport):
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_transport.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, None))
        mock_transport.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        async with MCPClient("http://localhost:8000/mcp") as client:
            assert client._session is not None

    @pytest.mark.asyncio
    @patch("mcp_playground.client.sse_client")
    @patch("mcp_playground.client.ClientSession")
    async def test_context_manager_sse_transport(self, mock_session_cls, mock_transport):
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_transport.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_transport.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        async with MCPClient("http://localhost:8000/sse", transport="sse") as client:
            assert client._session is not None


class TestMCPClientListTools:
    """Test list_tools method."""

    @pytest.mark.asyncio
    @patch("mcp_playground.client.streamable_http_client")
    @patch("mcp_playground.client.ClientSession")
    async def test_list_tools(self, mock_session_cls, mock_transport):
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_transport.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, None))
        mock_transport.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {
            "properties": {
                "param1": {"type": "string", "description": "A test parameter"}
            },
            "required": ["param1"],
        }

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        async with MCPClient("http://localhost:8000/mcp") as client:
            tools = await client.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        assert tools[0].description == "A test tool"
        assert len(tools[0].parameters) == 1
        assert tools[0].parameters[0].name == "param1"
        assert tools[0].parameters[0].parameter_type == "string"
        assert tools[0].parameters[0].required is True


class TestMCPClientCallTool:
    """Test call_tool method."""

    @pytest.mark.asyncio
    @patch("mcp_playground.client.streamable_http_client")
    @patch("mcp_playground.client.ClientSession")
    async def test_call_tool(self, mock_session_cls, mock_transport):
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_transport.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, None))
        mock_transport.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_content = MagicMock()
        mock_content.text = "result: 15"
        mock_content.model_dump_json = MagicMock(return_value='{"text": "result: 15"}')

        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_result.isError = False

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        async with MCPClient("http://localhost:8000/mcp") as client:
            result = await client.call_tool("calculator", arguments={"a": 10, "b": 5})

        assert result.content == "result: 15"
        assert result.error_code == 0
        mock_session.call_tool.assert_called_once_with("calculator", arguments={"a": 10, "b": 5})

    @pytest.mark.asyncio
    @patch("mcp_playground.client.streamable_http_client")
    @patch("mcp_playground.client.ClientSession")
    async def test_call_tool_error(self, mock_session_cls, mock_transport):
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_transport.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, None))
        mock_transport.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_content = MagicMock()
        mock_content.text = "Error: division by zero"

        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_result.isError = True

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        async with MCPClient("http://localhost:8000/mcp") as client:
            result = await client.call_tool("calculator", arguments={"a": 1, "b": 0})

        assert result.error_code == 1

    @pytest.mark.asyncio
    @patch("mcp_playground.client.streamable_http_client")
    @patch("mcp_playground.client.ClientSession")
    async def test_invoke_tool_backward_compat(self, mock_session_cls, mock_transport):
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_transport.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, None))
        mock_transport.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_content = MagicMock()
        mock_content.text = "ok"

        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_result.isError = False

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        async with MCPClient("http://localhost:8000/mcp") as client:
            result = await client.invoke_tool("test", {"key": "val"})

        assert result.error_code == 0
        mock_session.call_tool.assert_called_once_with("test", arguments={"key": "val"})


class TestMCPClientOneShot:
    """Test one-shot (non-context-manager) usage."""

    @pytest.mark.asyncio
    @patch("mcp_playground.client.streamable_http_client")
    @patch("mcp_playground.client.ClientSession")
    async def test_one_shot_list_tools(self, mock_session_cls, mock_transport):
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_transport.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, None))
        mock_transport.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tool.description = "Tool one"
        mock_tool.inputSchema = {"properties": {}, "required": []}

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        client = MCPClient("http://localhost:8000/mcp")
        tools = await client.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "tool1"


class TestToolInvocationResult:
    """Test ToolInvocationResult dataclass."""

    def test_basic_result(self):
        result = ToolInvocationResult(content="hello", error_code=0)
        assert result.content == "hello"
        assert result.error_code == 0
        assert result.structured_content is None
        assert result.raw_content == []

    def test_result_with_structured_content(self):
        result = ToolInvocationResult(
            content="hello",
            error_code=0,
            structured_content={"key": "value"},
        )
        assert result.structured_content == {"key": "value"}

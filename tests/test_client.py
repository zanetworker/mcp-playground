"""
Tests for the MCP SSE Client.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from mcp_playground import MCPClient, ToolDef, ToolParameter, ToolInvocationResult


class TestMCPClient(unittest.TestCase):
    """Test cases for the MCPClient class."""

    def setUp(self):
        """Set up test fixtures."""
        self.endpoint = "http://localhost:8000/sse"
        self.client = MCPClient(self.endpoint)

    def test_init_valid_endpoint(self):
        """Test initialization with a valid endpoint."""
        client = MCPClient(self.endpoint)
        self.assertEqual(client.endpoint, self.endpoint)

    def test_init_invalid_endpoint(self):
        """Test initialization with an invalid endpoint."""
        with self.assertRaises(ValueError):
            MCPClient("ftp://example.com/sse")

    @patch("mcp_playground.client.sse_client")
    @patch("mcp_playground.client.ClientSession")
    def test_list_tools(self, mock_session_class, mock_sse_client):
        """Test listing tools from the MCP endpoint."""
        # Set up mocks
        mock_streams = MagicMock()
        mock_sse_client.return_value.__aenter__.return_value = mock_streams
        
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        # Mock the list_tools response
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "A test parameter"
                }
            }
        }
        
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]
        mock_session.list_tools.return_value = mock_tools_result
        
        # Run the test
        loop = asyncio.get_event_loop()
        tools = loop.run_until_complete(self.client.list_tools())
        
        # Assertions
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "test_tool")
        self.assertEqual(tools[0].description, "A test tool")
        self.assertEqual(len(tools[0].parameters), 1)
        self.assertEqual(tools[0].parameters[0].name, "param1")
        self.assertEqual(tools[0].parameters[0].parameter_type, "string")
        self.assertEqual(tools[0].parameters[0].description, "A test parameter")

    @patch("mcp_playground.client.sse_client")
    @patch("mcp_playground.client.ClientSession")
    def test_invoke_tool(self, mock_session_class, mock_sse_client):
        """Test invoking a tool with parameters."""
        # Set up mocks
        mock_streams = MagicMock()
        mock_sse_client.return_value.__aenter__.return_value = mock_streams
        
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        # Mock the call_tool response
        mock_content_item = MagicMock()
        mock_content_item.model_dump_json.return_value = '{"result": "success"}'
        
        mock_result = MagicMock()
        mock_result.content = [mock_content_item]
        mock_result.isError = False
        
        mock_session.call_tool.return_value = mock_result
        
        # Run the test
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self.client.invoke_tool("test_tool", {"param1": "value1"})
        )
        
        # Assertions
        self.assertEqual(result.content, '{"result": "success"}')
        self.assertEqual(result.error_code, 0)
        
        # Verify the call_tool was called with the correct arguments
        mock_session.call_tool.assert_called_once_with(
            "test_tool", {"param1": "value1"}
        )


if __name__ == "__main__":
    unittest.main()

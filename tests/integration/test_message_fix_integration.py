"""
Integration tests for the message duplication fix across all LLM bridges.

These tests verify that the fix works correctly with real LLM bridge
implementations (OpenAI, Ollama, Anthropic, OpenRouter).
"""

import unittest
import pytest
import os
from unittest.mock import AsyncMock, MagicMock
from tests.fixtures.test_data import SAMPLE_CONVERSATIONS
from mcp_playground.client import MCPClient, ToolDef
from mcp_playground.llm_bridge.openai_bridge import OpenAIBridge
from mcp_playground.llm_bridge.ollama_bridge import OllamaBridge
from mcp_playground.llm_bridge.anthropic_bridge import AnthropicBridge
from mcp_playground.llm_bridge.openrouter_bridge import OpenRouterBridge


class MockMCPClient:
    """Mock MCP client for integration tests."""
    
    async def list_tools(self):
        return [ToolDef(name="test_tool", description="A test tool", parameters=[])]
    
    async def invoke_tool(self, tool_name, kwargs):
        from tests.mocks.mock_components import MockToolResult
        return MockToolResult("Mock tool result")


@pytest.mark.integration
class TestMessageFixIntegration(unittest.TestCase):
    """Integration tests for message duplication fix across all bridges."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_mcp_client = MockMCPClient()
        self.sample_conversation = SAMPLE_CONVERSATIONS["basic_conversation"].copy()
    
    def test_openai_bridge_message_fix(self):
        """Test message fix with OpenAI bridge."""
        # Skip if no API key available
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OpenAI API key not available")
        
        # Given: OpenAI bridge with mock MCP client
        bridge = OpenAIBridge(self.mock_mcp_client, os.getenv("OPENAI_API_KEY"), "gpt-3.5-turbo")
        
        # Mock the OpenAI client to avoid actual API calls in tests
        bridge.llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        bridge.llm_client.chat.completions.create.return_value = mock_response
        
        # When: Processing messages with new method
        import asyncio
        result = asyncio.run(bridge.process_messages(self.sample_conversation))
        
        # Then: Should work without message duplication
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("final_llm_response", result)
        
        # Verify submit_messages was called (not submit_query)
        bridge.llm_client.chat.completions.create.assert_called()
        call_args = bridge.llm_client.chat.completions.create.call_args
        self.assertIn("messages", call_args.kwargs)
        
        # Messages should match original conversation (no duplication)
        submitted_messages = call_args.kwargs["messages"]
        self.assertEqual(len(submitted_messages), len(self.sample_conversation))
    
    def test_ollama_bridge_message_fix(self):
        """Test message fix with Ollama bridge."""
        # Given: Ollama bridge with mock client
        bridge = OllamaBridge(self.mock_mcp_client, "llama2")
        
        # Mock the Ollama client
        bridge.llm_client = AsyncMock()
        mock_response = {"message": {"content": "Test response"}}
        bridge.llm_client.chat.return_value = mock_response
        
        # When: Processing messages
        import asyncio
        result = asyncio.run(bridge.process_messages(self.sample_conversation))
        
        # Then: Should work correctly
        self.assertIsNotNone(result)
        self.assertIn("final_llm_response", result)
        
        # Verify correct method was called
        bridge.llm_client.chat.assert_called()
        call_args = bridge.llm_client.chat.call_args
        self.assertIn("messages", call_args.kwargs)
        
        # No message duplication
        submitted_messages = call_args.kwargs["messages"]
        self.assertEqual(len(submitted_messages), len(self.sample_conversation))
    
    def test_anthropic_bridge_message_fix(self):
        """Test message fix with Anthropic bridge."""
        # Skip if no API key available
        if not os.getenv("ANTHROPIC_API_KEY"):
            self.skipTest("Anthropic API key not available")
        
        # Given: Anthropic bridge with mock client
        bridge = AnthropicBridge(self.mock_mcp_client, os.getenv("ANTHROPIC_API_KEY"))
        
        # Mock the Anthropic client
        bridge.llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Test response"
        mock_response.content[0].type = "text"
        bridge.llm_client.messages.create.return_value = mock_response
        
        # When: Processing messages
        import asyncio
        result = asyncio.run(bridge.process_messages(self.sample_conversation))
        
        # Then: Should work correctly
        self.assertIsNotNone(result)
        self.assertIn("final_llm_response", result)
        
        # Verify correct method was called
        bridge.llm_client.messages.create.assert_called()
        call_args = bridge.llm_client.messages.create.call_args
        self.assertIn("messages", call_args.kwargs)
        
        # No message duplication
        submitted_messages = call_args.kwargs["messages"]
        self.assertEqual(len(submitted_messages), len(self.sample_conversation))
    
    def test_openrouter_bridge_message_fix(self):
        """Test message fix with OpenRouter bridge."""
        # Skip if no API key available
        if not os.getenv("OPENROUTER_API_KEY"):
            self.skipTest("OpenRouter API key not available")
        
        # Given: OpenRouter bridge
        bridge = OpenRouterBridge(
            self.mock_mcp_client, 
            os.getenv("OPENROUTER_API_KEY"),
            "openai/gpt-3.5-turbo"
        )
        
        # Mock the OpenAI client (OpenRouter uses OpenAI-compatible API)
        bridge.llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        bridge.llm_client.chat.completions.create.return_value = mock_response
        
        # When: Processing messages
        import asyncio
        result = asyncio.run(bridge.process_messages(self.sample_conversation))
        
        # Then: Should work correctly
        self.assertIsNotNone(result)
        self.assertIn("final_llm_response", result)
        
        # Verify correct method was called
        bridge.llm_client.chat.completions.create.assert_called()
        call_args = bridge.llm_client.chat.completions.create.call_args
        self.assertIn("messages", call_args.kwargs)
        
        # No message duplication
        submitted_messages = call_args.kwargs["messages"]
        self.assertEqual(len(submitted_messages), len(self.sample_conversation))
    
    def test_consistent_behavior_across_bridges(self):
        """Test that all bridges behave consistently with the message fix."""
        # Given: All bridge types with mocked clients
        bridges = []
        
        # OpenAI Bridge
        if os.getenv("OPENAI_API_KEY"):
            openai_bridge = OpenAIBridge(self.mock_mcp_client, os.getenv("OPENAI_API_KEY"))
            openai_bridge.llm_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Response"
            mock_response.choices[0].message.tool_calls = None
            openai_bridge.llm_client.chat.completions.create.return_value = mock_response
            bridges.append(("OpenAI", openai_bridge))
        
        # Ollama Bridge
        ollama_bridge = OllamaBridge(self.mock_mcp_client, "llama2")
        ollama_bridge.llm_client = AsyncMock()
        ollama_bridge.llm_client.chat.return_value = {"message": {"content": "Response"}}
        bridges.append(("Ollama", ollama_bridge))
        
        # When: Processing same conversation with all bridges
        import asyncio
        results = []
        
        for name, bridge in bridges:
            try:
                result = asyncio.run(bridge.process_messages(self.sample_conversation))
                results.append((name, result))
            except Exception as e:
                self.fail(f"Bridge {name} failed: {e}")
        
        # Then: All bridges should return valid results
        self.assertGreater(len(results), 0, "No bridges were tested")
        
        for name, result in results:
            with self.subTest(bridge=name):
                self.assertIsNotNone(result, f"{name} bridge returned None")
                self.assertIsInstance(result, dict, f"{name} bridge didn't return dict")
                self.assertIn("final_llm_response", result, f"{name} bridge missing final response")
    
    def test_tool_calling_integration_across_bridges(self):
        """Test tool calling integration across different bridges."""
        # Given: Conversation that should trigger tool calling
        tool_conversation = SAMPLE_CONVERSATIONS["tool_conversation"].copy()
        
        # Test with OpenAI bridge (if available)
        if os.getenv("OPENAI_API_KEY"):
            bridge = OpenAIBridge(self.mock_mcp_client, os.getenv("OPENAI_API_KEY"))
            
            # Mock tool calling response
            bridge.llm_client = MagicMock()
            
            # First call - tool calling response
            tool_response = MagicMock()
            tool_response.choices = [MagicMock()]
            tool_response.choices[0].message = MagicMock()
            tool_response.choices[0].message.content = None
            tool_response.choices[0].message.tool_calls = [MagicMock()]
            tool_response.choices[0].message.tool_calls[0].function = MagicMock()
            tool_response.choices[0].message.tool_calls[0].function.name = "test_tool"
            tool_response.choices[0].message.tool_calls[0].function.arguments = "{}"
            
            # Second call - final response
            final_response = MagicMock()
            final_response.choices = [MagicMock()]
            final_response.choices[0].message = MagicMock()
            final_response.choices[0].message.content = "Final response with tool result"
            
            bridge.llm_client.chat.completions.create.side_effect = [tool_response, final_response]
            
            # When: Processing tool conversation
            import asyncio
            result = asyncio.run(bridge.process_messages(tool_conversation))
            
            # Then: Should handle tool calling correctly
            self.assertIsNotNone(result)
            self.assertIn("tool_call", result)
            self.assertIn("tool_result", result)
            self.assertIn("final_llm_response", result)
            
            # Should have made two calls (initial + final)
            self.assertEqual(bridge.llm_client.chat.completions.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
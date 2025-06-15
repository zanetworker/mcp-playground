"""
Unit tests for the tool calling workflow.

These tests verify that tool calling works correctly and that final
LLM responses are generated after tool execution.
"""

import unittest
import pytest
from tests.mocks.mock_components import MockLLMBridge, MockMCPClient, MockToolResult
from tests.fixtures.test_data import SAMPLE_CONVERSATIONS, SAMPLE_TOOL_CALLS


@pytest.mark.unit
class TestToolCallingWorkflow(unittest.TestCase):
    """Test cases for the tool calling workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MockMCPClient()
    
    def test_tool_execution_generates_final_response(self):
        """Test that tool execution generates a final LLM response."""
        # Given: A bridge configured to call tools
        bridge = MockLLMBridge(self.mock_client, should_call_tool=True)
        conversation = SAMPLE_CONVERSATIONS["tool_conversation"].copy()
        
        # When: Processing messages that should trigger tool calling
        import asyncio
        result = asyncio.run(bridge.process_messages(conversation))
        
        # Then: Tool should be called and final response generated
        self.assertTrue(bridge.tool_called, "Tool should have been called")
        self.assertTrue(bridge.final_response_generated, "Final response should have been generated")
        
        # Verify result structure
        self.assertIn("tool_call", result)
        self.assertIn("tool_result", result)
        self.assertIn("final_llm_response", result)
        
        # Verify final response exists
        self.assertIsNotNone(result["final_llm_response"])
        self.assertIsNotNone(result["tool_result"])
    
    def test_successful_tool_call_flow(self):
        """Test the complete flow: tool call → execution → final response."""
        # Given: A bridge that will call tools successfully
        bridge = MockLLMBridge(self.mock_client, should_call_tool=True, tool_should_fail=False)
        conversation = SAMPLE_CONVERSATIONS["tool_conversation"].copy()
        
        # When: Processing the conversation
        import asyncio
        result = asyncio.run(bridge.process_messages(conversation))
        
        # Then: Verify complete workflow
        # 1. Tool call was parsed
        tool_call = result.get("tool_call")
        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call["name"], "get_jira_issues")
        
        # 2. Tool was executed
        tool_result = result.get("tool_result")
        self.assertIsNotNone(tool_result)
        self.assertEqual(tool_result.error_code, 0)
        
        # 3. Final response was generated
        final_response = result.get("final_llm_response")
        self.assertIsNotNone(final_response)
        
        # 4. Both initial and final responses exist
        self.assertIn("llm_response", result)
        self.assertIn("final_llm_response", result)
    
    def test_failed_tool_call_handling(self):
        """Test behavior when tool execution fails."""
        # Given: A bridge where tools will fail
        bridge = MockLLMBridge(self.mock_client, should_call_tool=True, tool_should_fail=True)
        conversation = SAMPLE_CONVERSATIONS["tool_conversation"].copy()
        
        # When: Processing messages with failing tool
        import asyncio
        result = asyncio.run(bridge.process_messages(conversation))
        
        # Then: Tool failure should be handled gracefully
        self.assertTrue(bridge.tool_called, "Tool should have been called")
        
        # Tool result should indicate failure
        tool_result = result.get("tool_result")
        self.assertIsNotNone(tool_result)
        self.assertEqual(tool_result.error_code, 1)
        
        # Final response should still be the initial response (fallback)
        final_response = result.get("final_llm_response")
        initial_response = result.get("llm_response")
        self.assertEqual(final_response, initial_response)
    
    def test_no_tool_call_scenario(self):
        """Test when LLM doesn't call any tools."""
        # Given: A bridge that won't call tools
        bridge = MockLLMBridge(self.mock_client, should_call_tool=False)
        conversation = SAMPLE_CONVERSATIONS["basic_conversation"].copy()
        
        # When: Processing messages without tool calls
        import asyncio
        result = asyncio.run(bridge.process_messages(conversation))
        
        # Then: No tools should be called
        self.assertFalse(bridge.tool_called, "No tools should have been called")
        
        # Tool call and result should be None
        self.assertIsNone(result.get("tool_call"))
        self.assertIsNone(result.get("tool_result"))
        
        # Final response should be the same as initial response
        final_response = result.get("final_llm_response")
        initial_response = result.get("llm_response")
        self.assertEqual(final_response, initial_response)
    
    def test_tool_call_parsing(self):
        """Test that tool calls are correctly parsed from LLM responses."""
        # Given: A bridge that will call tools
        bridge = MockLLMBridge(self.mock_client, should_call_tool=True)
        
        # When: Parsing a mock LLM response with tool calls
        mock_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "get_jira_issues",
                            "arguments": '{"project": "TEST"}'
                        }
                    }]
                }
            }]
        }
        
        import asyncio
        parsed_call = asyncio.run(bridge.parse_tool_call(mock_response))
        
        # Then: Tool call should be correctly parsed
        self.assertIsNotNone(parsed_call)
        self.assertEqual(parsed_call["name"], "get_jira_issues")
        self.assertIn("parameters", parsed_call)
    
    def test_multiple_messages_with_tool_call(self):
        """Test tool calling in a multi-turn conversation."""
        # Given: A multi-turn conversation ending with a tool request
        conversation = SAMPLE_CONVERSATIONS["multi_turn_conversation"].copy()
        conversation.append({"role": "user", "content": "Get JIRA issues for project RHOAIREF"})
        
        bridge = MockLLMBridge(self.mock_client, should_call_tool=True)
        
        # When: Processing the conversation
        import asyncio
        result = asyncio.run(bridge.process_messages(conversation))
        
        # Then: Should handle the full conversation correctly
        self.assertEqual(len(bridge.submitted_messages), len(conversation))
        self.assertTrue(bridge.tool_called)
        self.assertIsNotNone(result.get("final_llm_response"))
    
    def test_tool_result_processing(self):
        """Test that tool results are properly processed for final response."""
        # Given: A bridge with tool calling enabled
        bridge = MockLLMBridge(self.mock_client, should_call_tool=True)
        conversation = SAMPLE_CONVERSATIONS["tool_conversation"].copy()
        
        # When: Processing messages
        import asyncio
        result = asyncio.run(bridge.process_messages(conversation))
        
        # Then: Tool result should be processed into final response
        self.assertTrue(bridge.final_response_generated)
        
        # Verify the process_tool_result was called by checking final response
        final_response = result.get("final_llm_response")
        self.assertIsNotNone(final_response)
        
        # Final response should be different from initial response (processed)
        initial_response = result.get("llm_response")
        self.assertNotEqual(final_response, initial_response)
    
    def test_original_query_extraction_from_messages(self):
        """Test extraction of original query from message history for tool processing."""
        # Given: A conversation with multiple messages
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Get JIRA issues for RHOAIREF"}
        ]
        
        bridge = MockLLMBridge(self.mock_client, should_call_tool=True)
        
        # When: Processing messages
        import asyncio
        result = asyncio.run(bridge.process_messages(conversation))
        
        # Then: The original query should be extracted correctly
        # (This is tested indirectly through successful tool processing)
        self.assertTrue(bridge.tool_called)
        self.assertTrue(bridge.final_response_generated)
        self.assertIsNotNone(result.get("final_llm_response"))


if __name__ == "__main__":
    unittest.main()
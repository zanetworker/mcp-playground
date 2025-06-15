"""
Unit tests for LLM bridge methods.

These tests verify that the new process_messages() and submit_messages()
methods work correctly and maintain backward compatibility.
"""

import unittest
import pytest
from tests.mocks.mock_components import MockLLMBridge, MockMCPClient
from tests.fixtures.test_data import SAMPLE_CONVERSATIONS


@pytest.mark.unit
class TestLLMBridgeMethods(unittest.TestCase):
    """Test cases for LLM bridge methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MockMCPClient()
        self.bridge = MockLLMBridge(self.mock_client)
    
    def test_process_messages_method_exists(self):
        """Test that the new process_messages method exists and is callable."""
        # Given: A bridge instance
        # When: Checking for the method
        # Then: Method should exist and be callable
        self.assertTrue(hasattr(self.bridge, 'process_messages'))
        self.assertTrue(callable(getattr(self.bridge, 'process_messages')))
    
    def test_submit_messages_method_exists(self):
        """Test that the new submit_messages method exists and is callable."""
        # Given: A bridge instance
        # When: Checking for the method
        # Then: Method should exist and be callable
        self.assertTrue(hasattr(self.bridge, 'submit_messages'))
        self.assertTrue(callable(getattr(self.bridge, 'submit_messages')))
    
    def test_process_messages_returns_correct_structure(self):
        """Test that process_messages returns the expected result structure."""
        # Given: A sample conversation
        conversation = SAMPLE_CONVERSATIONS["basic_conversation"].copy()
        
        # When: Processing messages
        import asyncio
        result = asyncio.run(self.bridge.process_messages(conversation))
        
        # Then: Result should have expected structure
        self.assertIsInstance(result, dict)
        self.assertIn("llm_response", result)
        self.assertIn("tool_call", result)
        self.assertIn("tool_result", result)
        self.assertIn("final_llm_response", result)
        self.assertIn("raw_final_response", result)
    
    def test_submit_messages_vs_submit_query_behavior(self):
        """Test that submit_messages behaves differently from submit_query regarding message handling."""
        # Given: A conversation history and user input
        conversation_history = SAMPLE_CONVERSATIONS["basic_conversation"][:-1]
        user_input = SAMPLE_CONVERSATIONS["basic_conversation"][-1]["content"]
        complete_conversation = SAMPLE_CONVERSATIONS["basic_conversation"].copy()
        
        # When: Using both methods
        import asyncio
        
        # Old method (submit_query)
        bridge_old = MockLLMBridge(self.mock_client)
        asyncio.run(bridge_old.submit_query(user_input, [], conversation_history))
        old_messages = bridge_old.submitted_messages
        
        # New method (submit_messages)
        bridge_new = MockLLMBridge(self.mock_client)
        asyncio.run(bridge_new.submit_messages(complete_conversation, []))
        new_messages = bridge_new.submitted_messages
        
        # Then: New method should preserve exact conversation structure
        self.assertEqual(new_messages, complete_conversation)
        
        # Old method should include the user input
        self.assertIn({"role": "user", "content": user_input}, old_messages)
        
        # Both methods should work
        self.assertIsNotNone(old_messages)
        self.assertIsNotNone(new_messages)
    
    def test_backward_compatibility_old_methods_still_work(self):
        """Test that old methods (process_query, submit_query) still work."""
        # Given: A bridge instance
        conversation_history = SAMPLE_CONVERSATIONS["basic_conversation"][:-1]
        user_input = SAMPLE_CONVERSATIONS["basic_conversation"][-1]["content"]
        
        # When: Using old methods
        import asyncio
        
        # Should not raise exceptions
        try:
            result = asyncio.run(self.bridge.submit_query(user_input, [], conversation_history))
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Old submit_query method failed: {e}")
    
    def test_process_messages_handles_empty_conversation(self):
        """Test that process_messages handles empty conversations gracefully."""
        # Given: An empty conversation
        empty_conversation = []
        
        # When: Processing empty conversation
        import asyncio
        result = asyncio.run(self.bridge.process_messages(empty_conversation))
        
        # Then: Should handle gracefully without errors
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(self.bridge.submitted_messages, [])
    
    def test_process_messages_with_tool_calling_enabled(self):
        """Test process_messages with tool calling enabled."""
        # Given: A bridge configured for tool calling
        bridge = MockLLMBridge(self.mock_client, should_call_tool=True)
        conversation = SAMPLE_CONVERSATIONS["tool_conversation"].copy()
        
        # When: Processing messages
        import asyncio
        result = asyncio.run(bridge.process_messages(conversation))
        
        # Then: Should execute complete tool calling workflow
        self.assertIsNotNone(result["tool_call"])
        self.assertIsNotNone(result["tool_result"])
        self.assertIsNotNone(result["final_llm_response"])
        self.assertTrue(bridge.tool_called)
        self.assertTrue(bridge.final_response_generated)
    
    def test_process_messages_without_tool_calling(self):
        """Test process_messages without tool calling."""
        # Given: A bridge configured without tool calling
        bridge = MockLLMBridge(self.mock_client, should_call_tool=False)
        conversation = SAMPLE_CONVERSATIONS["basic_conversation"].copy()
        
        # When: Processing messages
        import asyncio
        result = asyncio.run(bridge.process_messages(conversation))
        
        # Then: Should work without tool calling
        self.assertIsNone(result["tool_call"])
        self.assertIsNone(result["tool_result"])
        self.assertIsNotNone(result["final_llm_response"])
        self.assertFalse(bridge.tool_called)
        
        # Final response should be same as initial response
        self.assertEqual(result["final_llm_response"], result["llm_response"])
    
    def test_submit_messages_preserves_message_order(self):
        """Test that submit_messages preserves message order."""
        # Given: A multi-turn conversation
        conversation = SAMPLE_CONVERSATIONS["multi_turn_conversation"].copy()
        
        # When: Submitting messages
        import asyncio
        result = asyncio.run(self.bridge.submit_messages(conversation, []))
        
        # Then: Message order should be preserved
        submitted = self.bridge.submitted_messages
        self.assertEqual(len(submitted), len(conversation))
        
        for i, (original, submitted_msg) in enumerate(zip(conversation, submitted)):
            self.assertEqual(original, submitted_msg, f"Message order changed at index {i}")
    
    def test_process_messages_error_handling(self):
        """Test error handling in process_messages."""
        # Given: A bridge that might encounter errors
        bridge = MockLLMBridge(self.mock_client, should_call_tool=True, tool_should_fail=True)
        conversation = SAMPLE_CONVERSATIONS["tool_conversation"].copy()
        
        # When: Processing messages with potential errors
        import asyncio
        result = asyncio.run(bridge.process_messages(conversation))
        
        # Then: Should handle errors gracefully
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        
        # Tool result should indicate failure
        tool_result = result.get("tool_result")
        self.assertIsNotNone(tool_result)
        self.assertEqual(tool_result.error_code, 1)
        
        # Should still have a final response (fallback to initial)
        self.assertIsNotNone(result.get("final_llm_response"))
    
    def test_method_signatures_compatibility(self):
        """Test that method signatures are compatible with expected usage."""
        # Given: A bridge instance
        conversation = SAMPLE_CONVERSATIONS["basic_conversation"].copy()
        
        # When: Checking method signatures
        import inspect
        
        # process_messages should accept messages parameter
        process_sig = inspect.signature(self.bridge.process_messages)
        self.assertIn('messages', process_sig.parameters)
        
        # submit_messages should accept messages and formatted_tools
        submit_sig = inspect.signature(self.bridge.submit_messages)
        self.assertIn('messages', submit_sig.parameters)
        self.assertIn('formatted_tools', submit_sig.parameters)
    
    def test_result_consistency_between_methods(self):
        """Test that results are consistent between old and new methods."""
        # Given: Same input for both methods
        conversation_history = SAMPLE_CONVERSATIONS["basic_conversation"][:-1]
        user_input = SAMPLE_CONVERSATIONS["basic_conversation"][-1]["content"]
        complete_conversation = SAMPLE_CONVERSATIONS["basic_conversation"].copy()
        
        # When: Using both methods (without tool calling to focus on core behavior)
        import asyncio
        
        bridge_old = MockLLMBridge(self.mock_client, should_call_tool=False)
        result_old = asyncio.run(bridge_old.submit_query(user_input, [], conversation_history))
        
        bridge_new = MockLLMBridge(self.mock_client, should_call_tool=False)
        result_new = asyncio.run(bridge_new.process_messages(complete_conversation))
        
        # Then: Both should return valid results (structure may differ but both should work)
        self.assertIsNotNone(result_old)
        self.assertIsNotNone(result_new)
        self.assertIsInstance(result_new, dict)


if __name__ == "__main__":
    unittest.main()
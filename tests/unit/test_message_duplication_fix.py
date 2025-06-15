"""
Unit tests for the message duplication fix.

These tests verify that the new process_messages() method eliminates
message duplication that was present in the old process_query() method.
"""

import unittest
import pytest
from tests.mocks.mock_components import MockLLMBridge, MockMCPClient
from tests.fixtures.test_data import SAMPLE_CONVERSATIONS


@pytest.mark.unit
class TestMessageDuplicationFix(unittest.TestCase):
    """Test cases for the message duplication fix."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MockMCPClient()
        self.bridge = MockLLMBridge(self.mock_client)
    
    def test_no_duplicate_messages_in_process_messages(self):
        """Test that process_messages doesn't duplicate user messages."""
        # Given: A conversation with 3 messages
        conversation = SAMPLE_CONVERSATIONS["basic_conversation"].copy()
        original_length = len(conversation)
        
        # When: Processing messages with the new method
        import asyncio
        result = asyncio.run(self.bridge.process_messages(conversation))
        
        # Then: Messages sent to LLM should match original conversation exactly
        self.assertEqual(len(self.bridge.submitted_messages), original_length)
        self.assertEqual(self.bridge.submitted_messages, conversation)
        
        # Verify no duplication of the last user message
        user_messages = [msg for msg in self.bridge.submitted_messages if msg.get("role") == "user"]
        expected_user_messages = [msg for msg in conversation if msg.get("role") == "user"]
        self.assertEqual(len(user_messages), len(expected_user_messages))
    
    def test_old_vs_new_method_message_handling(self):
        """Test that new method handles messages differently from old method."""
        # Given: A conversation and user input
        conversation_history = SAMPLE_CONVERSATIONS["basic_conversation"][:-1]  # All but last message
        user_input = SAMPLE_CONVERSATIONS["basic_conversation"][-1]["content"]  # Last message content
        complete_conversation = SAMPLE_CONVERSATIONS["basic_conversation"].copy()
        
        # When: Using old method (submit_query)
        import asyncio
        bridge_old = MockLLMBridge(self.mock_client)
        result_old = asyncio.run(bridge_old.submit_query(user_input, [], conversation_history))
        old_messages = bridge_old.submitted_messages
        
        # When: Using new method (process_messages)
        bridge_new = MockLLMBridge(self.mock_client)
        result_new = asyncio.run(bridge_new.process_messages(complete_conversation))
        new_messages = bridge_new.submitted_messages
        
        # Then: New method should preserve exact conversation structure
        self.assertEqual(new_messages, complete_conversation)
        
        # Old method adds the user input to conversation history (potential duplication pattern)
        self.assertIn({"role": "user", "content": user_input}, old_messages)
        
        # Both methods should work, but new method is cleaner
        self.assertIsNotNone(result_old)
        self.assertIsNotNone(result_new)
    
    def test_conversation_history_integrity(self):
        """Test that conversation history remains intact and ordered."""
        # Given: A multi-turn conversation
        conversation = SAMPLE_CONVERSATIONS["multi_turn_conversation"].copy()
        
        # When: Processing with new method
        import asyncio
        result = asyncio.run(self.bridge.process_messages(conversation))
        
        # Then: Conversation order and content should be preserved
        submitted = self.bridge.submitted_messages
        self.assertEqual(len(submitted), len(conversation))
        
        for i, (original, submitted_msg) in enumerate(zip(conversation, submitted)):
            self.assertEqual(original["role"], submitted_msg["role"], f"Role mismatch at index {i}")
            self.assertEqual(original["content"], submitted_msg["content"], f"Content mismatch at index {i}")
    
    def test_empty_conversation_handling(self):
        """Test handling of empty conversation."""
        # Given: An empty conversation
        conversation = SAMPLE_CONVERSATIONS["empty_conversation"]
        
        # When: Processing empty conversation
        import asyncio
        result = asyncio.run(self.bridge.process_messages(conversation))
        
        # Then: Should handle gracefully without errors
        self.assertEqual(self.bridge.submitted_messages, [])
        self.assertIsNotNone(result)
    
    def test_single_message_conversation(self):
        """Test handling of single message conversation."""
        # Given: A conversation with just one message
        conversation = SAMPLE_CONVERSATIONS["single_message"].copy()
        
        # When: Processing single message
        import asyncio
        result = asyncio.run(self.bridge.process_messages(conversation))
        
        # Then: Should preserve the single message without duplication
        self.assertEqual(len(self.bridge.submitted_messages), 1)
        self.assertEqual(self.bridge.submitted_messages[0], conversation[0])
    
    def test_original_query_extraction_for_tool_processing(self):
        """Test that original query is correctly extracted for tool result processing."""
        # Given: A conversation ending with a user message
        conversation = SAMPLE_CONVERSATIONS["tool_conversation"].copy()
        bridge_with_tools = MockLLMBridge(self.mock_client, should_call_tool=True)
        
        # When: Processing messages that trigger tool calling
        import asyncio
        result = asyncio.run(bridge_with_tools.process_messages(conversation))
        
        # Then: Tool should be called and final response generated
        self.assertTrue(bridge_with_tools.tool_called)
        self.assertTrue(bridge_with_tools.final_response_generated)
        self.assertIsNotNone(result.get("final_llm_response"))
    
    def test_message_format_consistency(self):
        """Test that message format is consistent between old and new methods."""
        # Given: A conversation
        conversation = SAMPLE_CONVERSATIONS["basic_conversation"].copy()
        
        # When: Processing with new method
        import asyncio
        result = asyncio.run(self.bridge.process_messages(conversation))
        
        # Then: All messages should have proper format
        for msg in self.bridge.submitted_messages:
            self.assertIn("role", msg)
            self.assertIn("content", msg)
            self.assertIn(msg["role"], ["user", "assistant", "system"])
            self.assertIsInstance(msg["content"], str)


if __name__ == "__main__":
    unittest.main()
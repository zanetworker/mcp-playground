"""
Test fixtures and sample data for MCP playground tests.
"""

from typing import Dict, List, Any

# Sample conversations for testing
SAMPLE_CONVERSATIONS = {
    "basic_conversation": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ],
    
    "tool_conversation": [
        {"role": "user", "content": "Get me the JIRA issues for RHOAIREF project"}
    ],
    
    "multi_turn_conversation": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
        {"role": "user", "content": "Can you search for files?"},
        {"role": "assistant", "content": "I can help you search for files. What are you looking for?"},
        {"role": "user", "content": "Find Python files in the project"}
    ],
    
    "empty_conversation": [],
    
    "single_message": [
        {"role": "user", "content": "What's the weather like?"}
    ]
}

# Sample tool calls for testing
SAMPLE_TOOL_CALLS = {
    "jira_issues": {
        "id": "call_1",
        "name": "get_jira_issues",
        "parameters": {
            "project": "RHOAIREF",
            "component": "RAG_AGENTIC",
            "label": "RAG",
            "issuetype": "feature",
            "max_results": 100,
            "format": "json"
        }
    },
    
    "search_files": {
        "id": "call_2", 
        "name": "search_files",
        "parameters": {
            "pattern": "*.py",
            "directory": "/src"
        }
    },
    
    "invalid_tool": {
        "id": "call_3",
        "name": "nonexistent_tool",
        "parameters": {}
    }
}

# Sample LLM responses for testing
SAMPLE_LLM_RESPONSES = {
    "simple_response": {
        "choices": [{
            "message": {
                "content": "This is a simple response without tool calls."
            }
        }]
    },
    
    "tool_call_response": {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_jira_issues",
                        "arguments": '{"project": "RHOAIREF", "component": "RAG_AGENTIC"}'
                    }
                }]
            }
        }]
    },
    
    "final_response_with_tool_result": {
        "choices": [{
            "message": {
                "content": "Based on the JIRA search, I found 3 issues in the RHOAIREF project: RHOAIREF-5464, RHOAIREF-5465, and RHOAIREF-5466. These are all feature requests related to RAG and Agentic workflows."
            }
        }]
    },
    
    "error_response": {
        "error": {
            "message": "API rate limit exceeded",
            "type": "rate_limit_error"
        }
    }
}

# Sample tool results for testing
SAMPLE_TOOL_RESULTS = {
    "successful_jira_result": {
        "content": "Found 3 issues:\n**RHOAIREF-5464**: Model Selection and Optimization for RAG/Agentic Workflows\n**RHOAIREF-5465**: Develop a Comprehensive Evaluation Framework for RAG and Agentic Systems\n**RHOAIREF-5466**: Introduce Product Metrics and Observability Metrics for OpenShift AI RAG/Agentic Components",
        "error_code": 0
    },
    
    "failed_jira_result": {
        "content": "Error: Unable to connect to JIRA API. Please check your credentials.",
        "error_code": 1
    },
    
    "successful_file_search": {
        "content": "Found 15 Python files:\n- src/main.py\n- src/utils.py\n- tests/test_main.py\n...",
        "error_code": 0
    },
    
    "empty_result": {
        "content": "No results found.",
        "error_code": 0
    }
}

# Edge cases for testing
EDGE_CASES = {
    "very_long_conversation": [
        {"role": "user", "content": f"Message {i}"} 
        for i in range(100)
    ],
    
    "malformed_messages": [
        {"role": "user"},  # Missing content
        {"content": "Hello"},  # Missing role
        {"role": "invalid", "content": "Test"}  # Invalid role
    ],
    
    "unicode_content": [
        {"role": "user", "content": "Hello 🌍! How are you? 中文测试 العربية"},
        {"role": "assistant", "content": "I can handle Unicode! 😊 🚀 ✨"}
    ]
}
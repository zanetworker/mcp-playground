"""
Pytest configuration and shared fixtures for MCP playground tests.
"""

import pytest
from typing import Dict, List, Any
from tests.mocks.mock_components import MockMCPClient, MockToolResult
from tests.fixtures.test_data import SAMPLE_CONVERSATIONS, SAMPLE_TOOL_CALLS


@pytest.fixture
def mock_mcp_client():
    """Provide a mock MCP client for testing."""
    return MockMCPClient()


@pytest.fixture
def sample_conversation():
    """Provide a sample conversation for testing."""
    return SAMPLE_CONVERSATIONS["basic_conversation"]


@pytest.fixture
def sample_tool_conversation():
    """Provide a conversation that includes tool calling."""
    return SAMPLE_CONVERSATIONS["tool_conversation"]


@pytest.fixture
def sample_tool_call():
    """Provide a sample tool call for testing."""
    return SAMPLE_TOOL_CALLS["jira_issues"]


@pytest.fixture
def successful_tool_result():
    """Provide a successful tool result for testing."""
    return MockToolResult(
        content="Found 3 issues: RHOAIREF-5464, RHOAIREF-5465, RHOAIREF-5466",
        error_code=0
    )


@pytest.fixture
def failed_tool_result():
    """Provide a failed tool result for testing."""
    return MockToolResult(
        content="Error: Unable to connect to JIRA API",
        error_code=1
    )


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (slower, requires external dependencies)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
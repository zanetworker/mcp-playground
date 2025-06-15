"""
Mock components for testing MCP playground functionality.
"""

from typing import Dict, List, Any, Optional
from mcp_playground.llm_bridge.base import LLMBridge
from mcp_playground.client import ToolDef, ToolInvocationResult


class MockToolResult:
    """Mock tool result for testing."""
    
    def __init__(self, content: str, error_code: int = 0):
        self.content = content
        self.error_code = error_code


class MockMCPClient:
    """Mock MCP client for testing."""
    
    def __init__(self):
        self.tools = [
            ToolDef(name="get_jira_issues", description="Get JIRA issues", parameters=[]),
            ToolDef(name="search_files", description="Search files", parameters=[])
        ]
    
    async def list_tools(self) -> List[ToolDef]:
        """Return mock tools."""
        return self.tools
    
    async def invoke_tool(self, tool_name: str, kwargs: Dict[str, Any]) -> ToolInvocationResult:
        """Mock tool invocation."""
        if tool_name == "get_jira_issues":
            return MockToolResult("Found 3 issues: RHOAIREF-5464, RHOAIREF-5465, RHOAIREF-5466")
        elif tool_name == "search_files":
            return MockToolResult("Found 5 files matching criteria")
        else:
            return MockToolResult("Unknown tool", error_code=1)


class MockLLMBridge(LLMBridge):
    """Configurable mock LLM bridge for testing."""
    
    def __init__(self, mcp_client=None, should_call_tool=False, tool_should_fail=False):
        super().__init__(mcp_client or MockMCPClient())
        self.should_call_tool = should_call_tool
        self.tool_should_fail = tool_should_fail
        self.submitted_messages = None
        self.submitted_query = None
        self.tool_called = False
        self.final_response_generated = False
    
    async def format_tools(self, tools: List[ToolDef]) -> List[Dict[str, Any]]:
        """Return mock formatted tools."""
        return [{"name": tool.name, "description": tool.description} for tool in tools]
    
    async def submit_query(self, query: str, formatted_tools: Any, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Mock the old submit_query method."""
        self.submitted_query = query
        messages = conversation_history.copy() if conversation_history else []
        messages.append({"role": "user", "content": query})
        self.submitted_messages = messages
        return self._generate_mock_response()
    
    async def submit_messages(self, messages: List[Dict[str,str]], formatted_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock the new submit_messages method."""
        self.submitted_messages = messages
        return self._generate_mock_response()
    
    def _generate_mock_response(self) -> Dict[str, Any]:
        """Generate a mock LLM response."""
        if self.should_call_tool:
            return {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_jira_issues",
                                "arguments": '{"project": "RHOAIREF"}'
                            }
                        }]
                    }
                }]
            }
        else:
            return {
                "choices": [{
                    "message": {
                        "content": "This is a mock LLM response without tool calls."
                    }
                }]
            }
    
    async def parse_tool_call(self, llm_response: Any) -> Optional[Dict[str, Any]]:
        """Mock tool call parsing."""
        if self.should_call_tool and "tool_calls" in str(llm_response):
            return {
                "id": "call_1",
                "name": "get_jira_issues",
                "parameters": {"project": "RHOAIREF"}
            }
        return None
    
    async def submit_query_without_tools(self, messages: List[Dict[str, Any]]) -> Any:
        """Mock final response generation."""
        self.final_response_generated = True
        return {
            "choices": [{
                "message": {
                    "content": "Based on the tool results, here is the final response."
                }
            }]
        }
    
    async def execute_tool(self, tool_name: str, kwargs: Dict[str, Any]) -> ToolInvocationResult:
        """Mock tool execution."""
        self.tool_called = True
        if self.tool_should_fail:
            return MockToolResult("Tool execution failed", error_code=1)
        else:
            return MockToolResult("Tool executed successfully")
    
    async def process_tool_result(self, original_query: str, tool_call: Dict[str, Any], tool_result: Any, conversation_history: Optional[List[Dict[str, str]]] = None) -> Any:
        """Mock tool result processing."""
        return await self.submit_query_without_tools([])
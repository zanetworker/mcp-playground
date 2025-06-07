"""
Base class for LLM Bridge implementations.
"""
import abc
from typing import Dict, List, Any, Optional
from ..client import MCPClient, ToolDef, ToolInvocationResult


class LLMBridge(abc.ABC):
    """Abstract base class for LLM bridge implementations."""
    
    def __init__(self, mcp_client: MCPClient):
        """Initialize the LLM bridge with an MCPClient instance.
        
        Args:
            mcp_client: An initialized MCPClient instance
        """
        self.mcp_client = mcp_client
        self.tools = None
    
    async def fetch_tools(self) -> List[ToolDef]:
        """Fetch available tools from the MCP endpoint.
        
        Returns:
            List of ToolDef objects
        """
        self.tools = await self.mcp_client.list_tools()
        return self.tools
    
    @abc.abstractmethod
    async def format_tools(self, tools: List[ToolDef]) -> Any:
        """Format tools for the specific LLM provider.
        
        Args:
            tools: List of ToolDef objects
            
        Returns:
            Formatted tools in the LLM-specific format
        """
        pass
    
    @abc.abstractmethod
    async def submit_query(self, query: str, formatted_tools: Any, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Submit a query to the LLM with the formatted tools.
        
        Args:
            query: User query string
            formatted_tools: Tools in the LLM-specific format
            conversation_history: Previous conversation messages (optional)
            
        Returns:
            LLM response
        """
        pass
    
    @abc.abstractmethod
    async def parse_tool_call(self, llm_response: Any) -> Optional[Dict[str, Any]]:
        """Parse the LLM response to extract tool calls.
        
        Args:
            llm_response: Response from the LLM
            
        Returns:
            Dictionary with tool name and parameters, or None if no tool call
        """
        pass
    
    async def execute_tool(self, tool_name: str, kwargs: Dict[str, Any]) -> ToolInvocationResult:
        """Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to invoke
            kwargs: Dictionary of parameters to pass to the tool
            
        Returns:
            ToolInvocationResult containing the tool's response
        """
        return await self.mcp_client.invoke_tool(tool_name, kwargs)
    
    async def process_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Process a user query through the LLM and execute any tool calls.
        
        This method handles the full flow:
        1. Fetch tools if not already fetched
        2. Format tools for the LLM
        3. Submit query to LLM
        4. Parse tool calls from LLM response
        5. Execute tool if needed
        
        Args:
            query: User query string
            conversation_history: Previous conversation messages (optional)
            
        Returns:
            Dictionary containing the LLM response, tool call, and tool result
        """
        # 1. Fetch tools if not already fetched
        if self.tools is None:
            await self.fetch_tools()
        
        # 2. Format tools for the LLM
        formatted_tools = await self.format_tools(self.tools)
        
        # 3. Submit query to LLM
        llm_response = await self.submit_query(query, formatted_tools, conversation_history)
        
        # 4. Parse tool calls from LLM response
        tool_call = await self.parse_tool_call(llm_response)
        
        result = {
            "llm_response": llm_response,
            "tool_call": tool_call,
            "tool_result": None
        }
        
        # 5. Execute tool if needed
        if tool_call:
            tool_name = tool_call.get("name")
            kwargs = tool_call.get("parameters", {})
            tool_result = await self.execute_tool(tool_name, kwargs)
            result["tool_result"] = tool_result
        
        return result

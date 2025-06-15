"""
Anthropic-specific implementation of the LLM Bridge.
"""
from typing import Dict, List, Any, Optional
import anthropic
from ..client import ToolDef
from ..format_converters import to_anthropic_format
from .base import LLMBridge
from .models import DEFAULT_ANTHROPIC_MODEL # Import default model


class AnthropicBridge(LLMBridge):
    """Anthropic-specific implementation of the LLM Bridge."""
    
    def __init__(self, mcp_client, api_key, model=DEFAULT_ANTHROPIC_MODEL): # Use imported default
        """Initialize Anthropic bridge with API key and model.
        
        Args:
            mcp_client: An initialized MCPClient instance
            api_key: Anthropic API key
            model: Anthropic model to use (default: from models.py)
        """
        super().__init__(mcp_client)
        self.llm_client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Store provider info for metadata
        self.provider_info = {
            "provider": "anthropic",
            "model": model,
            "base_url": "https://api.anthropic.com"
        }
    
    async def format_tools(self, tools: List[ToolDef]) -> List[Dict[str, Any]]:
        """Format tools for Anthropic.
        
        Args:
            tools: List of ToolDef objects
            
        Returns:
            List of tools in Anthropic format
        """
        return to_anthropic_format(tools)
    
    async def submit_query(self, query: str, formatted_tools: List[Dict[str, Any]], conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Submit a query to Anthropic with the formatted tools.
        
        Args:
            query: User query string
            formatted_tools: Tools in Anthropic format
            conversation_history: Previous conversation messages (optional)
            
        Returns:
            Anthropic API response
        """
        # Build messages with conversation history
        messages = conversation_history.copy() if conversation_history else []
        messages.append({"role": "user", "content": query})
        
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=4096,
            system="You are a helpful tool-using assistant.",
            messages=messages,
            tools=formatted_tools
        )
        
        return response
    
    async def submit_query_without_tools(self, messages: List[Dict[str, Any]]) -> Any:
        """Submit a query to Anthropic without tools for final processing.
        
        Args:
            messages: Complete conversation including tool results
            
        Returns:
            Anthropic API response
        """
        # Make the API call without tools
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=messages
            # Note: No tools parameter - this is for final processing
        )
        
        return response
    
    async def submit_messages(self, messages: List[Dict[str,str]], formatted_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Submit messages to Anthropic with the formatted tools.
        
        Args:
            messages: List of message dictionaries
            formatted_tools: Tools in Anthropic format
            
        Returns:
            Anthropic API response
        """
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=messages,
            tools=formatted_tools
        )
        
        return response
    
    async def parse_tool_call(self, llm_response: Any) -> Optional[Dict[str, Any]]:
        """Parse the Anthropic response to extract tool calls.
        
        Args:
            llm_response: Response from Anthropic
            
        Returns:
            Dictionary with tool name and parameters, or None if no tool call
        """
        for content in llm_response.content:
            if content.type == "tool_use":
                return {
                    "name": content.name,  # Access name directly from the ToolUseBlock
                    "parameters": content.input  # Access input directly from the ToolUseBlock
                }
        
        return None

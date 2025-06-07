"""
OpenRouter-based implementation of the LLM Bridge for unified provider access.
"""
from typing import Dict, List, Any, Optional
import openai
from ..client import ToolDef
from ..format_converters import to_openai_format
from .base import LLMBridge
from .openrouter_client import OpenRouterClient


class OpenRouterBridge(LLMBridge):
    """OpenRouter-based implementation of the LLM Bridge."""
    
    def __init__(self, mcp_client, api_key: str, model: str, site_url: Optional[str] = None, site_name: Optional[str] = None):
        """Initialize OpenRouter bridge.
        
        Args:
            mcp_client: An initialized MCPClient instance
            api_key: OpenRouter API key
            model: Model ID (e.g., 'openai/gpt-4o', 'anthropic/claude-3-opus')
            site_url: Optional site URL for rankings
            site_name: Optional site name for rankings
        """
        super().__init__(mcp_client)
        self.api_key = api_key
        self.model = model
        self.site_url = site_url
        self.site_name = site_name
        
        # Initialize OpenAI client with OpenRouter base URL
        self.llm_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        # Initialize OpenRouter client for model fetching
        self.openrouter_client = OpenRouterClient(api_key, site_url, site_name)
        
        # Store provider info for metadata
        self.provider_info = {
            "provider": "openrouter",
            "model": model,
            "base_url": "https://openrouter.ai/api/v1"
        }
    
    async def format_tools(self, tools: List[ToolDef]) -> List[Dict[str, Any]]:
        """Format tools for OpenRouter (uses OpenAI format).
        
        Args:
            tools: List of ToolDef objects
            
        Returns:
            List of tools in OpenAI format (compatible with OpenRouter)
        """
        return to_openai_format(tools)
    
    async def submit_query(self, query: str, formatted_tools: List[Dict[str, Any]], conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Submit a query to OpenRouter with the formatted tools.
        
        Args:
            query: User query string
            formatted_tools: Tools in OpenAI format
            conversation_history: Previous conversation messages (optional)
            
        Returns:
            OpenRouter API response (OpenAI-compatible format)
        """
        # Build messages with conversation history
        messages = conversation_history.copy() if conversation_history else []
        messages.append({"role": "user", "content": query})
        
        # Prepare extra headers for OpenRouter
        extra_headers = self.openrouter_client.get_extra_headers()
        
        # Make the API call
        if formatted_tools:
            response = self.llm_client.chat.completions.create(
                extra_headers=extra_headers,
                model=self.model,
                messages=messages,
                tools=formatted_tools,
                tool_choice="auto"
            )
        else:
            response = self.llm_client.chat.completions.create(
                extra_headers=extra_headers,
                model=self.model,
                messages=messages
            )
        
        return response
    
    async def submit_query_without_tools(self, messages: List[Dict[str, Any]]) -> Any:
        """Submit a query to OpenRouter without tools for final processing.
        
        Args:
            messages: Complete conversation including tool results
            
        Returns:
            OpenRouter API response (OpenAI-compatible format)
        """
        # Prepare extra headers for OpenRouter
        extra_headers = self.openrouter_client.get_extra_headers()
        
        # Make the API call without tools
        response = self.llm_client.chat.completions.create(
            extra_headers=extra_headers,
            model=self.model,
            messages=messages
            # Note: No tools parameter - this is for final processing
        )
        
        return response
    
    async def parse_tool_call(self, llm_response: Any) -> Optional[Dict[str, Any]]:
        """Parse the OpenRouter response to extract tool calls.
        
        Args:
            llm_response: Response from OpenRouter (OpenAI-compatible format)
            
        Returns:
            Dictionary with tool name and parameters, or None if no tool call
        """
        if hasattr(llm_response, 'choices') and llm_response.choices:
            choice = llm_response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                tool_call = choice.message.tool_calls[0]
                if hasattr(tool_call, 'function'):
                    try:
                        import json
                        return {
                            "name": tool_call.function.name,
                            "parameters": json.loads(tool_call.function.arguments)
                        }
                    except (json.JSONDecodeError, AttributeError):
                        pass
        
        return None
    
    async def get_available_models(self, provider: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top available models for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'google')
            limit: Maximum number of models to return
            
        Returns:
            List of top models for the provider
        """
        return await self.openrouter_client.fetch_top_models_by_provider(provider, limit)
"""
Ollama-specific implementation of the LLM Bridge for local models.
"""
from typing import Dict, List, Any, Optional
import json
import ollama
from ..client import ToolDef
from ..format_converters import to_openai_format # Ollama uses OpenAI-like tool format
from .base import LLMBridge
from .models import OPENAI_MODELS # Re-use OpenAI format for tools

# Note: Ollama model names are user-defined (e.g., 'llama3', 'mistral')
# We won't define a static list here, but allow users to specify.
DEFAULT_OLLAMA_MODEL = "llama3" # A common default, user might need to change

class OllamaBridge(LLMBridge):
    """Ollama-specific implementation of the LLM Bridge."""
    
    def __init__(self, mcp_client, model=DEFAULT_OLLAMA_MODEL, host=None):
        """Initialize Ollama bridge with model and optional host.
        
        Args:
            mcp_client: An initialized MCPClient instance.
            model: Ollama model name to use (e.g., 'llama3', 'mistral').
                   Ensure the model is available locally in Ollama.
            host: Optional URL of the Ollama server (e.g., 'http://localhost:11434').
                  If None, the default host configured for the ollama library will be used.
        """
        super().__init__(mcp_client)
        # Initialize Ollama client, optionally specifying the host
        self.llm_client = ollama.AsyncClient(host=host) 
        self.model = model
        self.host = host # Store host for potential display/debugging
        print(f"Ollama Bridge initialized. Model: {self.model}, Host: {self.host or 'default'}")

    async def format_tools(self, tools: List[ToolDef]) -> List[Dict[str, Any]]:
        """Format tools for Ollama (uses OpenAI-like format).
        
        Args:
            tools: List of ToolDef objects
            
        Returns:
            List of tools in Ollama/OpenAI format
        """
        return to_openai_format(tools)
    
    async def submit_query(self, query: str, formatted_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Submit a query to Ollama with the formatted tools.
        
        Args:
            query: User query string
            formatted_tools: Tools in Ollama/OpenAI format
            
        Returns:
            Ollama API response object (dictionary-like)
        """
        messages = [{"role": "user", "content": query}]
        
        try:
            response = await self.llm_client.chat(
                model=self.model,
                messages=messages,
                tools=formatted_tools,
                # Ollama automatically decides on tool use if tools are provided
            )
            # The response object is already a dictionary
            return response 
        except ollama.ResponseError as e:
            # Handle potential errors like model not found
            print(f"Ollama API Error: {e.error} (Status code: {e.status_code})")
            # Re-raise or return an error structure if needed
            raise e 
        except Exception as e:
            print(f"An unexpected error occurred with Ollama: {e}")
            raise e

    async def parse_tool_call(self, llm_response: Any) -> Optional[Dict[str, Any]]:
        """Parse the Ollama response to extract tool calls.
        
        Args:
            llm_response: Response dictionary from Ollama
            
        Returns:
            Dictionary with tool name and parameters, or None if no tool call
        """
        message = llm_response.get('message', {})
        tool_calls = message.get('tool_calls')

        if not tool_calls:
            return None
        
        # Ollama might return multiple tool calls, handle the first one for now
        tool_call = tool_calls[0] 
        function_info = tool_call.get('function', {})
        
        # Ensure arguments are loaded as JSON if they are a string
        arguments = function_info.get('arguments', {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse tool arguments as JSON: {arguments}")
                arguments = {} # Fallback to empty dict

        return {
            "name": function_info.get('name'),
            "parameters": arguments
        }

    async def check_connection(self):
        """Check if the Ollama server is reachable and the model exists."""
        try:
            # Simple check to see if server responds
            await self.llm_client.list()
            
            # Check if the specific model exists locally
            models_info = await self.llm_client.list()
            
            # Debug the response structure
            print(f"Ollama models response in check_connection: {models_info}")
            
            # Handle different response structures
            model_names = []
            if isinstance(models_info, dict) and 'models' in models_info:
                # New API format
                model_names = [m.get('name', m.get('model', '')) for m in models_info.get('models', [])]
            elif isinstance(models_info, list):
                # Older API format or direct list
                model_names = [m.get('name', m.get('model', '')) for m in models_info]
            
            # Filter out empty names
            model_names = [name for name in model_names if name]
            
            if not model_names:
                print("Warning: No models found in Ollama response")
                return True  # Still return True as the server is reachable
                
            if self.model not in model_names:
                print(f"Warning: Model '{self.model}' not found in local Ollama models: {model_names}")
                # Depending on strictness, could raise an error here
            return True
        except Exception as e:
            print(f"Error connecting to Ollama host '{self.host or 'default'}': {e}")
            return False

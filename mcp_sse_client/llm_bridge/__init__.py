"""
LLM Bridge for integrating MCP client with various LLM providers.
"""

from .base import LLMBridge
from .openai_bridge import OpenAIBridge
from .anthropic_bridge import AnthropicBridge
from .ollama_bridge import OllamaBridge
from .openrouter_bridge import OpenRouterBridge
from .openrouter_client import OpenRouterClient, format_model_display

__all__ = [
    "LLMBridge",
    "OpenAIBridge",
    "AnthropicBridge",
    "OllamaBridge",
    "OpenRouterBridge",
    "OpenRouterClient",
    "format_model_display"
]

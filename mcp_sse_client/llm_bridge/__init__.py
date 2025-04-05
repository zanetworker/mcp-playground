"""
LLM Bridge for integrating MCP client with various LLM providers.
"""
"""LLM Bridge module for integrating MCP with various LLM providers."""

from .base import LLMBridge
from .openai_bridge import OpenAIBridge
from .anthropic_bridge import AnthropicBridge
from .ollama_bridge import OllamaBridge # Import OllamaBridge

__all__ = [
    "LLMBridge", 
    "OpenAIBridge", 
    "AnthropicBridge", 
    "OllamaBridge" # Add OllamaBridge to __all__
]

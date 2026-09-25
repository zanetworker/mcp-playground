"""
Defines available models and defaults for LLM providers.
Updated March 2026.
"""

OPENAI_MODELS = [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
    'o3',
    'o3-mini',
    'o4-mini',
]
DEFAULT_OPENAI_MODEL = 'gpt-4o'

ANTHROPIC_MODELS = [
    'claude-sonnet-4-20250514',
    'claude-opus-4-20250918',
    'claude-haiku-4-20250414',
    'claude-3-7-sonnet-20250219',
    'claude-3-5-sonnet-20241022',
    'claude-3-5-haiku-20241022',
]
DEFAULT_ANTHROPIC_MODEL = 'claude-sonnet-4-20250514'

DEFAULT_OLLAMA_MODEL = "llama3.3"

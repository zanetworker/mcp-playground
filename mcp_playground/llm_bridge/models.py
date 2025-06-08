"""
Defines available models and defaults for LLM providers.
"""

# OpenAI Models (Updated as of May 2025)
OPENAI_MODELS = [
    'gpt-4o', 
    'gpt-4.5-turbo',
    'gpt-4.5-preview',
    'gpt-4o-mini', 
    'gpt-4-turbo',
    'gpt-4-vision-preview',
    'gpt-3.5-turbo',
    'o1-preview',
    'o1-mini'
]
DEFAULT_OPENAI_MODEL = 'gpt-4o'

# Anthropic Models (Updated as of May 2025)
ANTHROPIC_MODELS = [
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307',
    'claude-3.5-sonnet-20250501',
    'claude-3-5-sonnet-20250501',
    'claude-3-7-sonnet-20250219',
    'claude-3-5-haiku-20241022-v1:0',
    'claude-2.1',
    'claude-2.0'
]
DEFAULT_ANTHROPIC_MODEL = 'claude-3-5-sonnet-20250501'

# Ensure defaults are in the lists
if DEFAULT_OPENAI_MODEL not in OPENAI_MODELS:
    # Fallback if default is somehow removed or renamed
    DEFAULT_OPENAI_MODEL = OPENAI_MODELS[0] if OPENAI_MODELS else 'gpt-4o' 

if DEFAULT_ANTHROPIC_MODEL not in ANTHROPIC_MODELS:
    # Fallback
    DEFAULT_ANTHROPIC_MODEL = ANTHROPIC_MODELS[0] if ANTHROPIC_MODELS else 'claude-3-7-sonnet-20250219'

# Ollama Default Model
# Note: Ollama models are installed locally by the user (e.g., 'llama3', 'mistral').
# We don't maintain a list here, but define a common default.
DEFAULT_OLLAMA_MODEL = "llama3"

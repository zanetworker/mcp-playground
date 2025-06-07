import streamlit as st
import sys
import os
import json
import asyncio
import datetime
import traceback
import ollama  # Import ollama for model listing
from typing import List, Dict, Any, Optional, Union

# Add parent directory to path to import mcp_sse_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the MCP SSE client and model definitions
from mcp_sse_client.client import MCPClient, MCPConnectionError, MCPTimeoutError
from mcp_sse_client.llm_bridge.openai_bridge import OpenAIBridge
from mcp_sse_client.llm_bridge.anthropic_bridge import AnthropicBridge
from mcp_sse_client.llm_bridge.ollama_bridge import OllamaBridge
from mcp_sse_client.llm_bridge.openrouter_bridge import OpenRouterBridge
from mcp_sse_client.llm_bridge.openrouter_client import OpenRouterClient, format_model_display
from mcp_sse_client.llm_bridge.models import (
    OPENAI_MODELS, DEFAULT_OPENAI_MODEL,
    ANTHROPIC_MODELS, DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OLLAMA_MODEL
)

# Set page config
st.set_page_config(
    page_title="MCP Tool Tester",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply custom styling
load_css()

# Initialize session state
if "connected" not in st.session_state:
    st.session_state.connected = False
if "client" not in st.session_state:
    st.session_state.client = None
if "llm_bridge" not in st.session_state:
    st.session_state.llm_bridge = None
if "tools" not in st.session_state:
    st.session_state.tools = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "connection_error" not in st.session_state:
    st.session_state.connection_error = None
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "openrouter": os.environ.get("OPENROUTER_API_KEY", "")
    }
if "mcp_endpoint" not in st.session_state:
    st.session_state.mcp_endpoint = "http://localhost:8001/sse"
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "openai"
if "openai_model" not in st.session_state:
    st.session_state.openai_model = DEFAULT_OPENAI_MODEL
if "anthropic_model" not in st.session_state:
    st.session_state.anthropic_model = DEFAULT_ANTHROPIC_MODEL
if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = DEFAULT_OLLAMA_MODEL
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = ""
if "ollama_models" not in st.session_state:
    st.session_state.ollama_models = []
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "auto"  # auto, chat, tools
if "show_tools_only" not in st.session_state:
    st.session_state.show_tools_only = True  # Default to showing only tool-capable models

# Auto-refresh session state
if "models_loaded_on_startup" not in st.session_state:
    st.session_state.models_loaded_on_startup = False
if "last_provider" not in st.session_state:
    st.session_state.last_provider = None
if "auto_refresh_enabled" not in st.session_state:
    st.session_state.auto_refresh_enabled = True

# OpenRouter session state
if "openrouter_site_url" not in st.session_state:
    st.session_state.openrouter_site_url = os.environ.get("OPENROUTER_SITE_URL", "")
if "openrouter_site_name" not in st.session_state:
    st.session_state.openrouter_site_name = os.environ.get("OPENROUTER_SITE_NAME", "")

# OpenRouter model caches for each provider
if "openai_openrouter_models" not in st.session_state:
    st.session_state.openai_openrouter_models = []
if "anthropic_openrouter_models" not in st.session_state:
    st.session_state.anthropic_openrouter_models = []
if "google_openrouter_models" not in st.session_state:
    st.session_state.google_openrouter_models = []

# Selected OpenRouter models
if "openai_openrouter_model" not in st.session_state:
    st.session_state.openai_openrouter_model = None
if "anthropic_openrouter_model" not in st.session_state:
    st.session_state.anthropic_openrouter_model = None
if "google_openrouter_model" not in st.session_state:
    st.session_state.google_openrouter_model = None

# --- Ollama Helper Functions ---
async def fetch_ollama_models(host=None):
    """Asynchronously fetch available Ollama models from the server.
    
    Args:
        host: Optional Ollama host URL. If None, uses default.
        
    Returns:
        List of model names, or empty list if error occurs.
    """
    try:
        client = ollama.AsyncClient(host=host)
        models_info = await client.list()
        
        # Extract model names from the response
        model_names = []
        
        # Direct extraction from ListResponse object (most common case)
        if hasattr(models_info, 'models'):
            # Extract directly from the models attribute
            for model in models_info.models:
                if hasattr(model, 'model') and model.model:
                    model_names.append(model.model)
        
        # If we couldn't extract models directly and model_names is still empty
        if not model_names:
            # Try parsing the string representation as a fallback
            models_str = str(models_info)
            
            # Use regex to extract model names from the string representation
            import re
            pattern = r"model='([^']+)'"
            model_names = re.findall(pattern, models_str)
            
            # If that didn't work, try other formats
            if not model_names and isinstance(models_info, dict) and 'models' in models_info:
                # Dictionary format
                model_names = [m.get('name', m.get('model', '')) for m in models_info.get('models', [])]
            elif not model_names and isinstance(models_info, list):
                # List format
                model_names = [m.get('name', m.get('model', '')) for m in models_info]
        
        # Filter out empty names
        model_names = [name for name in model_names if name]
        
        return model_names
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []

# --- Tool Capability Detection ---
def is_tool_capable_model(model_id: str, model_data: dict = None) -> bool:
    """Determine if a model supports tool/function calling.
    
    Args:
        model_id: The model identifier (e.g., 'openai/gpt-4')
        model_data: Optional model metadata from OpenRouter API
        
    Returns:
        bool: True if the model supports tools, False otherwise
    """
    # Check OpenRouter metadata first if available
    if model_data:
        # Check for explicit tool support flags
        supports_tools = model_data.get("supports_tools", False)
        supports_function_calling = model_data.get("supports_function_calling", False)
        if supports_tools or supports_function_calling:
            return True
    
    # Fallback to pattern matching for known tool-capable models
    model_lower = model_id.lower()
    
    # OpenAI models with tool support
    if any(pattern in model_lower for pattern in [
        "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"
    ]):
        # Exclude base models and instruct variants that typically don't support tools
        if any(exclude in model_lower for exclude in ["base", "instruct"]):
            return False
        return True
    
    # Anthropic Claude models with tool support
    if any(pattern in model_lower for pattern in [
        "claude-3", "claude-3.5"
    ]):
        return True
    
    # Google Gemini models with tool support
    if any(pattern in model_lower for pattern in [
        "gemini-1.5", "gemini-pro"
    ]):
        # Exclude vision-only models
        if "vision" in model_lower and "pro" not in model_lower:
            return False
        return True
    
    # Meta models with tool support
    if any(pattern in model_lower for pattern in [
        "llama-3.1", "llama-3.2"
    ]):
        # Only larger models typically support tools
        if any(size in model_lower for size in ["70b", "405b"]):
            return True
    
    # Mistral models with tool support
    if any(pattern in model_lower for pattern in [
        "mistral-large", "mistral-medium", "mixtral"
    ]):
        return True
    
    # Default to False for unknown models
    return False

# --- OpenRouter Helper Functions ---
async def fetch_openrouter_models_by_provider(api_key, provider, limit=5, tools_only=False):
    """Fetch top N most popular models for a specific provider from OpenRouter.
    
    Args:
        api_key: OpenRouter API key
        provider: Provider name (e.g., 'openai', 'anthropic', 'google')
        limit: Maximum number of models to return
        tools_only: If True, only return models that support tool calling
        
    Returns:
        List of formatted model dictionaries
    """
    try:
        client = OpenRouterClient(
            api_key=api_key,
            site_url=st.session_state.openrouter_site_url,
            site_name=st.session_state.openrouter_site_name
        )
        
        # Fetch more models initially if filtering for tools
        fetch_limit = limit * 3 if tools_only else limit
        models = await client.fetch_top_models_by_provider(provider, fetch_limit)
        
        # Filter for tool-capable models if requested
        if tools_only:
            tool_capable_models = []
            for model in models:
                if is_tool_capable_model(model["id"], model):
                    tool_capable_models.append(model)
                    if len(tool_capable_models) >= limit:
                        break
            models = tool_capable_models
        
        # Format models for display
        formatted_models = []
        for model in models:
            formatted = format_model_display(model, include_tool_indicator=True)
            formatted_models.append(formatted)
        
        return formatted_models
    except Exception as e:
        print(f"Error fetching {provider} models from OpenRouter: {e}")
        return []

def sync_fetch_openrouter_models(api_key, provider, limit=5, tools_only=False):
    """Synchronous wrapper for OpenRouter model fetching."""
    try:
        # Handle event loop properly
        try:
            loop = asyncio.get_running_loop()
            # If we're in an existing loop, use thread executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    fetch_openrouter_models_by_provider(api_key, provider, limit, tools_only)
                )
                return future.result(timeout=30)
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(fetch_openrouter_models_by_provider(api_key, provider, limit, tools_only))
    except Exception as e:
        print(f"Error in sync fetch: {e}")
        return []

# --- Auto-Refresh Functions ---
async def auto_refresh_models_async(provider: str, force: bool = False) -> bool:
    """Auto-refresh models for the specified provider.
    
    Args:
        provider: The LLM provider ('openai', 'anthropic', 'google', 'ollama')
        force: Force refresh even if models are already loaded
        
    Returns:
        bool: True if models were successfully refreshed, False otherwise
    """
    try:
        # Check if we should refresh
        if not force and not should_refresh_models(provider):
            return True
        
        if provider in ["openai", "anthropic", "google"]:
            # OpenRouter providers
            api_key = st.session_state.api_keys.get("openrouter")
            if not api_key:
                print(f"No OpenRouter API key available for {provider}")
                return False
            
            models = await fetch_openrouter_models_by_provider(
                api_key, provider, 5, st.session_state.show_tools_only
            )
            
            if models:
                st.session_state[f"{provider}_openrouter_models"] = models
                # Auto-select first model if none selected
                selected_model_key = f"{provider}_openrouter_model"
                if not st.session_state.get(selected_model_key):
                    st.session_state[selected_model_key] = models[0]["id"]
                return True
            else:
                print(f"No models found for {provider}")
                return False
                
        elif provider == "ollama":
            # Ollama provider
            models = await fetch_ollama_models(st.session_state.ollama_host)
            if models:
                st.session_state.ollama_models = [str(model) for model in models]
                # Auto-select first model if current model not in list
                if (st.session_state.ollama_model not in st.session_state.ollama_models and
                    st.session_state.ollama_models):
                    st.session_state.ollama_model = st.session_state.ollama_models[0]
                return True
            else:
                print("No Ollama models found")
                return False
        
        return False
        
    except Exception as e:
        print(f"Error auto-refreshing {provider} models: {e}")
        return False

def auto_refresh_models(provider: str, force: bool = False) -> bool:
    """Synchronous wrapper for auto-refresh models."""
    try:
        # Handle event loop properly
        try:
            loop = asyncio.get_running_loop()
            # If we're in an existing loop, use thread executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, auto_refresh_models_async(provider, force))
                return future.result(timeout=30)
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(auto_refresh_models_async(provider, force))
    except Exception as e:
        print(f"Error in sync auto-refresh: {e}")
        return False

def should_refresh_models(provider: str) -> bool:
    """Determine if models should be refreshed for the given provider."""
    if provider in ["openai", "anthropic", "google"]:
        models_key = f"{provider}_openrouter_models"
        return not st.session_state.get(models_key)
    elif provider == "ollama":
        return not st.session_state.get("ollama_models")
    return True

def detect_provider_change() -> bool:
    """Detect if the provider has changed since last run."""
    current_provider = st.session_state.llm_provider
    last_provider = st.session_state.get("last_provider")
    
    if last_provider != current_provider:
        st.session_state.last_provider = current_provider
        return True
    return False

def handle_startup_auto_refresh():
    """Handle auto-refresh on app startup."""
    if not st.session_state.models_loaded_on_startup and st.session_state.auto_refresh_enabled:
        current_provider = st.session_state.llm_provider
        
        with st.spinner(f"Loading {current_provider} models..."):
            success = auto_refresh_models(current_provider, force=False)
            if success:
                st.session_state.models_loaded_on_startup = True
                st.session_state.last_provider = current_provider

def handle_provider_change_auto_refresh():
    """Handle auto-refresh when provider changes."""
    if detect_provider_change() and st.session_state.auto_refresh_enabled:
        current_provider = st.session_state.llm_provider
        
        with st.spinner(f"Loading {current_provider} models..."):
            success = auto_refresh_models(current_provider, force=True)
            if success:
                # Clear previous provider's selection to avoid confusion
                providers = ["openai", "anthropic", "google"]
                for provider in providers:
                    if provider != current_provider:
                        selected_key = f"{provider}_openrouter_model"
                        if selected_key in st.session_state:
                            st.session_state[selected_key] = None

# --- Direct LLM Chat Functions ---
async def chat_with_llm_directly(user_input):
    """Chat directly with LLM without tools."""
    if st.session_state.llm_provider == "ollama":
        try:
            host = st.session_state.ollama_host if st.session_state.ollama_host else None
            client = ollama.AsyncClient(host=host)
            
            response = await client.chat(
                model=st.session_state.ollama_model,
                messages=[{"role": "user", "content": user_input}]
            )
            
            return response.get('message', {}).get('content', 'No response received')
            
        except Exception as e:
            return f"Error chatting with Ollama: {e}"
    
    elif st.session_state.llm_provider in ["openai", "anthropic", "google"]:
        try:
            # Use OpenRouter for these providers
            import openai
            
            # Get selected model for the provider
            selected_model_key = f"{st.session_state.llm_provider}_openrouter_model"
            selected_model = st.session_state.get(selected_model_key)
            
            if not selected_model:
                return f"No {st.session_state.llm_provider} model selected. Please select a model first."
            
            if not st.session_state.api_keys["openrouter"]:
                return "No OpenRouter API key configured. Please add your API key."
            
            # Create OpenRouter client
            client = openai.AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=st.session_state.api_keys["openrouter"]
            )
            
            # Prepare extra headers
            extra_headers = {}
            if st.session_state.openrouter_site_url:
                extra_headers["HTTP-Referer"] = st.session_state.openrouter_site_url
            if st.session_state.openrouter_site_name:
                extra_headers["X-Title"] = st.session_state.openrouter_site_name
            
            response = await client.chat.completions.create(
                extra_headers=extra_headers,
                model=selected_model,
                messages=st.session_state.messages + [{"role": "user", "content": user_input}]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error chatting with {st.session_state.llm_provider} via OpenRouter: {e}"
    
    return "No LLM provider configured for direct chat."

# --- Connection Functions ---
async def connect_to_server_async():
    """Connect to MCP server and LLM provider with enhanced error handling."""
    try:
        st.session_state.connection_error = None
        
        # Create MCP client with correct parameters (no retry_delay)
        client = MCPClient(
            st.session_state.mcp_endpoint,
            timeout=30.0,
            max_retries=3
        )
        
        # Test connection by listing tools
        tools = await client.list_tools()
        
        # Create LLM bridge based on provider
        llm_bridge = None
        if st.session_state.llm_provider in ["openai", "anthropic", "google"] and st.session_state.api_keys["openrouter"]:
            # Get selected model for the provider
            selected_model_key = f"{st.session_state.llm_provider}_openrouter_model"
            selected_model = st.session_state.get(selected_model_key)
            
            if selected_model:
                llm_bridge = OpenRouterBridge(
                    client,
                    api_key=st.session_state.api_keys["openrouter"],
                    model=selected_model,
                    site_url=st.session_state.openrouter_site_url,
                    site_name=st.session_state.openrouter_site_name
                )
        elif st.session_state.llm_provider == "ollama":
            host = st.session_state.ollama_host if st.session_state.ollama_host else None
            llm_bridge = OllamaBridge(client, model=st.session_state.ollama_model, host=host)
        
        # Update session state
        st.session_state.client = client
        st.session_state.llm_bridge = llm_bridge
        st.session_state.tools = tools
        st.session_state.connected = True
        
        return True, f"Connected to {st.session_state.mcp_endpoint}", len(tools)
            
    except MCPConnectionError as e:
        st.session_state.connected = False
        st.session_state.connection_error = f"Connection failed: {e}"
        return False, f"Connection failed: {e}", 0
    except MCPTimeoutError as e:
        st.session_state.connected = False
        st.session_state.connection_error = f"Connection timed out: {e}"
        return False, f"Connection timed out: {e}", 0
    except Exception as e:
        st.session_state.connected = False
        st.session_state.connection_error = f"Unexpected error: {e}"
        return False, f"Unexpected error: {e}\n{traceback.format_exc()}", 0

def connect_to_server():
    """Synchronous wrapper for async connection function."""
    try:
        # Handle event loop properly
        try:
            loop = asyncio.get_running_loop()
            # If we're in an existing loop, use thread executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, connect_to_server_async())
                success, message, tool_count = future.result(timeout=60)
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            success, message, tool_count = asyncio.run(connect_to_server_async())
        
        if success:
            st.success(f"✅ {message}")
            if st.session_state.llm_bridge:
                st.success(f"✅ LLM bridge configured for {st.session_state.llm_provider}")
                if st.session_state.llm_provider == "ollama":
                    st.success(f"✅ Using Ollama model: {st.session_state.ollama_model}")
            if tool_count > 0:
                st.success(f"✅ Found {tool_count} available tools")
                # Show tool names for debugging
                # Removed duplicate "Available tools" display - keeping only the expandable dropdown version
                # if st.session_state.tools:
                #     tool_names = [tool.name for tool in st.session_state.tools]
                #     st.info(f"🔧 Available tools: {', '.join(tool_names)}")
            else:
                st.warning("⚠️ No tools found on the server")
        else:
            st.error(f"❌ {message}")
    except Exception as e:
        st.error(f"❌ Connection error: {e}")
        st.session_state.connected = False
        st.session_state.connection_error = str(e)

def disconnect_from_server():
    """Disconnect from server and clean up."""
    st.session_state.connected = False
    st.session_state.client = None
    st.session_state.llm_bridge = None
    st.session_state.tools = []
    st.session_state.connection_error = None
    st.success("✅ Disconnected from server")

# --- Response Parsing Helper ---
def extract_content_from_llm_response(llm_response, response_stage="final"):
    """Extract clean text content from different LLM provider response formats.
    
    Args:
        llm_response: Response object from OpenAI, Anthropic, Ollama, or dict
        response_stage: "initial" or "final" to handle different processing stages
        
    Returns:
        str: Clean text content extracted from the response (never None)
    """
    try:
        print(f"DEBUG extract_content_from_llm_response: Input type: {type(llm_response)}")
        print(f"DEBUG extract_content_from_llm_response: Input value: {llm_response}")
        
        # OpenAI ChatCompletion object
        if hasattr(llm_response, 'choices') and llm_response.choices:
            content = llm_response.choices[0].message.content
            print(f"DEBUG: OpenAI content extracted: {content}")
            return content if content is not None else "No content received from OpenAI"
        
        # Anthropic Message object
        elif hasattr(llm_response, 'content') and llm_response.content:
            for content in llm_response.content:
                if hasattr(content, 'type') and content.type == "text":
                    text = content.text
                    print(f"DEBUG: Anthropic text content extracted: {text}")
                    return text if text is not None else "No text content from Anthropic"
            # Fallback: return first content item as string
            first_content = str(llm_response.content[0])
            print(f"DEBUG: Anthropic fallback content: {first_content}")
            return first_content if first_content else "Empty content from Anthropic"
        
        # Dict response (could be from any provider)
        elif isinstance(llm_response, dict):
            print(f"DEBUG: Dict response keys: {list(llm_response.keys())}")
            
            # Check for direct content key
            if 'content' in llm_response:
                content = llm_response['content']
                print(f"DEBUG: Direct content key found: {content}")
                return content if content else "Empty content in dict"
            
            # Check for message.content (Ollama format)
            if 'message' in llm_response:
                message = llm_response['message']
                if isinstance(message, dict) and 'content' in message:
                    content = message['content']
                    print(f"DEBUG: Ollama message content: {content}")
                    return content if content is not None else "No content in Ollama message"
            
            # Check for response key (some providers use this)
            if 'response' in llm_response:
                response = llm_response['response']
                print(f"DEBUG: Response key found: {response}")
                return response if response else "Empty response in dict"
            
            # Check for text key
            if 'text' in llm_response:
                text = llm_response['text']
                print(f"DEBUG: Text key found: {text}")
                return text if text else "Empty text in dict"
            
            # Fallback: convert entire dict to string
            fallback = str(llm_response)
            print(f"DEBUG: Dict fallback: {fallback}")
            return fallback if fallback else "Empty dict response"
        
        # String response (already clean)
        elif isinstance(llm_response, str):
            print(f"DEBUG: String response: {llm_response}")
            return llm_response if llm_response else "Empty string response"
        
        # None response
        elif llm_response is None:
            print("DEBUG: None response")
            return "No response received"
        
        # Fallback: convert to string
        fallback = str(llm_response)
        print(f"DEBUG: Final fallback: {fallback}")
        return fallback if fallback else "Empty response object"
        
    except Exception as e:
        error_msg = f"Error extracting content: {e}"
        print(f"DEBUG: Exception in extract_content_from_llm_response: {error_msg}")
        return error_msg

def format_tool_result(tool_result_content):
    """Format tool result content for better readability.
    
    Args:
        tool_result_content: Raw tool result content (string)
        
    Returns:
        str: Formatted content for display
    """
    try:
        # Try to parse as JSON and format nicely
        if isinstance(tool_result_content, str) and tool_result_content.strip().startswith(('[', '{')):
            try:
                parsed_json = json.loads(tool_result_content)
                
                # Handle nested structure like {"type": "text", "text": "[...]"}
                if isinstance(parsed_json, dict) and 'text' in parsed_json:
                    # Extract the inner text content
                    inner_text = parsed_json['text']
                    if isinstance(inner_text, str) and inner_text.strip().startswith('['):
                        # Parse the inner JSON array
                        try:
                            inner_parsed = json.loads(inner_text)
                            parsed_json = inner_parsed
                        except json.JSONDecodeError:
                            # If inner parsing fails, use the text as-is
                            return inner_text
                
                # Special formatting for JIRA issues (common case)
                if isinstance(parsed_json, list) and len(parsed_json) > 0:
                    if all(isinstance(item, dict) and 'key' in item for item in parsed_json):
                        # Format as JIRA issues
                        formatted_issues = []
                        for issue in parsed_json:
                            issue_text = f"**{issue.get('key', 'Unknown')}**: {issue.get('summary', 'No summary')}"
                            if 'status' in issue:
                                issue_text += f"\n  - Status: {issue['status']}"
                            if 'priority' in issue:
                                issue_text += f"\n  - Priority: {issue['priority']}"
                            if 'assignee' in issue:
                                issue_text += f"\n  - Assignee: {issue['assignee']}"
                            if 'created' in issue:
                                issue_text += f"\n  - Created: {issue['created']}"
                            formatted_issues.append(issue_text)
                        
                        return f"Found {len(parsed_json)} issues:\n\n" + "\n\n".join(formatted_issues)
                
                # General JSON formatting
                return json.dumps(parsed_json, indent=2, ensure_ascii=False)
                
            except json.JSONDecodeError:
                # Not valid JSON, return as-is
                pass
        
        # Return original content if not JSON or formatting fails
        return tool_result_content
        
    except Exception as e:
        # If anything goes wrong, return original content
        return tool_result_content

# --- Chat Functions ---
async def process_user_message_async(user_input):
    """Process user message based on chat mode."""
    
    # Chat mode: direct LLM conversation without tools
    if st.session_state.chat_mode == "chat":
        return await chat_with_llm_directly(user_input)
    
    # Tools mode: always use tools
    elif st.session_state.chat_mode == "tools":
        if not st.session_state.llm_bridge:
            return "❌ No LLM bridge configured. Please configure an API key and connect to MCP server."
        
        try:
            result = await st.session_state.llm_bridge.process_query(user_input, st.session_state.messages)
            
            # Handle enhanced response structure for tools mode
            if isinstance(result, dict):
                # Extract responses from enhanced structure
                final_llm_response = result.get("final_llm_response", {})
                tool_call = result.get("tool_call")
                tool_result = result.get("tool_result")
                
                # Extract the final LLM content
                final_content = extract_content_from_llm_response(final_llm_response, "final")
                
                # Store enhanced response data for UI display
                enhanced_response_data = {
                    "final_llm_content": final_content,
                    "initial_llm_response": result.get("initial_llm_response", {}),
                    "final_llm_response": final_llm_response,
                    "raw_initial_response": result.get("raw_initial_response", {}),
                    "raw_final_response": result.get("raw_final_response", {}),
                    "tool_call": tool_call,
                    "tool_result": tool_result,
                    "processing_steps": result.get("processing_steps", []),
                    "metadata": result.get("metadata", {}),
                    "has_tools": hasattr(st.session_state.llm_bridge, 'tools') and st.session_state.llm_bridge.tools
                }
                
                # Store in session state for the UI to access
                st.session_state.last_response_data = enhanced_response_data
                
                response_parts = []
                
                # Only add content if it's not a generic "no content" message and we have a tool result
                if tool_call and tool_result:
                    # If we have a tool result, prioritize that over generic "no content" messages
                    if final_content and not final_content.startswith("No content received from"):
                        response_parts.append(final_content)
                    
                    tool_name = tool_call.get('name', 'Unknown')
                    if tool_result.error_code == 0:
                        response_parts.append(f"🔧 **Tool Used:** {tool_name}")
                        formatted_result = format_tool_result(tool_result.content)
                        response_parts.append(f"**Result:**\n{formatted_result}")
                    else:
                        response_parts.append(f"❌ **Tool Error:** {tool_name} failed")
                        response_parts.append(f"**Error:** {tool_result.content}")
                else:
                    # No tool call, just return the content
                    response_parts.append(final_content)
                
                return "\n".join(response_parts)
            else:
                # Handle legacy response format
                return extract_content_from_llm_response(result)
                
        except Exception as e:
            return f"Sorry, I encountered an error: {e}"
    
    # Auto mode: let LLM decide whether to use tools
    else:  # auto mode
        if not st.session_state.llm_bridge:
            # Fall back to direct chat if no bridge available
            return await chat_with_llm_directly(user_input)
        
        try:
            # Debug: Check if we have tools available (but don't show misleading warnings)
            if hasattr(st.session_state.llm_bridge, 'tools') and st.session_state.llm_bridge.tools:
                tools_count = len(st.session_state.llm_bridge.tools)
                # Only show this info in debug mode, not always
                # st.info(f"🔧 Auto mode: {tools_count} MCP tools available for LLM to use")
            
            result = await st.session_state.llm_bridge.process_query(user_input, st.session_state.messages)
            
            # Handle enhanced response structure
            if isinstance(result, dict):
                # Extract responses from enhanced structure
                initial_llm_response = result.get("initial_llm_response", {})
                final_llm_response = result.get("final_llm_response", {})
                raw_initial_response = result.get("raw_initial_response", {})
                raw_final_response = result.get("raw_final_response", {})
                tool_call = result.get("tool_call")
                tool_result = result.get("tool_result")
                processing_steps = result.get("processing_steps", [])
                metadata = result.get("metadata", {})
                
                # Extract the final LLM content (this is what the user should see)
                final_content = extract_content_from_llm_response(final_llm_response, "final")
                
                # Debug logging for enhanced structure
                print(f"DEBUG: Enhanced Response Structure:")
                print(f"  - Initial Response Type: {type(initial_llm_response)}")
                print(f"  - Final Response Type: {type(final_llm_response)}")
                print(f"  - Final Content: {final_content}")
                print(f"  - Tool Call: {tool_call}")
                print(f"  - Tool Result: {tool_result}")
                print(f"  - Processing Steps: {len(processing_steps)}")
                print(f"  - Metadata: {metadata}")
                
                # Store comprehensive response data for UI display
                enhanced_response_data = {
                    "final_llm_content": final_content,
                    "initial_llm_response": initial_llm_response,
                    "final_llm_response": final_llm_response,
                    "raw_initial_response": raw_initial_response,
                    "raw_final_response": raw_final_response,
                    "tool_call": tool_call,
                    "tool_result": tool_result,
                    "processing_steps": processing_steps,
                    "metadata": metadata,
                    "has_tools": hasattr(st.session_state.llm_bridge, 'tools') and st.session_state.llm_bridge.tools
                }
                
                # Store in session state for the UI to access
                st.session_state.last_response_data = enhanced_response_data
                
                # Return the final LLM content (this is the key fix!)
                if final_content and not final_content.startswith("No content received from") and not final_content.startswith("Error extracting content"):
                    return final_content
                else:
                    # Fallback handling for edge cases
                    if tool_call and tool_result and tool_result.error_code == 0:
                        return f"I successfully executed the {tool_call.get('name', 'requested')} tool and processed the results. Please check the details below."
                    else:
                        return "I processed your request using the available tools."
            else:
                # Handle legacy response format
                return extract_content_from_llm_response(result)
                
        except Exception as e:
            return f"Sorry, I encountered an error: {e}"

def process_user_message(user_input):
    """Synchronous wrapper for async message processing."""
    try:
        # Check if there's already an event loop running
        try:
            loop = asyncio.get_running_loop()
            # If we're in an existing loop, we need to use a different approach
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, process_user_message_async(user_input))
                result = future.result(timeout=60)  # 60 second timeout
            return result
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            result = asyncio.run(process_user_message_async(user_input))
            return result
    except Exception as e:
        return f"Error processing message: {e}"

# Modern Header
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 class="gradient-text" style="font-size: 3.5rem; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: -0.02em;">
        🛠️ MCP Tool Tester
    </h1>
    <p style="color: var(--text-secondary); font-size: 1.2rem; margin: 0;">
        Modern AI Tool Integration Platform
    </p>
    <div style="width: 100px; height: 3px; background: var(--accent-gradient); margin: 1rem auto; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 class="gradient-text" style="font-size: 1.8rem; margin-bottom: 0.5rem;">⚙️ Configuration</h2>
        <div style="width: 60px; height: 2px; background: var(--accent-gradient); margin: 0.5rem auto; border-radius: 1px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin: 1.5rem 0 1rem 0;">
        <h3 style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
            📡 Server Connection
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # MCP Endpoint
    mcp_endpoint = st.text_input(
        "MCP Endpoint URL",
        value=st.session_state.mcp_endpoint,
        help="Enter the full URL for the MCP SSE server."
    )
    st.session_state.mcp_endpoint = mcp_endpoint

    st.markdown("""
    <div style="margin: 1.5rem 0 1rem 0;">
        <h3 style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
            🤖 LLM Configuration
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # LLM Provider
    llm_provider = st.selectbox(
        "LLM Provider",
        ["openai", "anthropic", "google", "ollama"],
        index=["openai", "anthropic", "google", "ollama"].index(st.session_state.llm_provider) if st.session_state.llm_provider in ["openai", "anthropic", "google", "ollama"] else 0
    )
    st.session_state.llm_provider = llm_provider
    
    # Handle auto-refresh on startup and provider changes
    handle_startup_auto_refresh()
    handle_provider_change_auto_refresh()
    
    # Provider specific settings
    if llm_provider in ["openai", "anthropic", "google"]:
        st.markdown(f"""
        <div style="margin: 1.5rem 0 1rem 0;">
            <h3 style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                🎯 {llm_provider.title()} Models (via OpenRouter)
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # OpenRouter API Key
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=st.session_state.api_keys["openrouter"],
            help="Get your API key from openrouter.ai"
        )
        st.session_state.api_keys["openrouter"] = api_key
        
        # Optional site information
        with st.expander("Optional: Site Information for Rankings"):
            site_url = st.text_input(
                "Site URL",
                value=st.session_state.openrouter_site_url,
                help="Your site URL for rankings on openrouter.ai"
            )
            st.session_state.openrouter_site_url = site_url
            
            site_name = st.text_input(
                "Site Name",
                value=st.session_state.openrouter_site_name,
                help="Your site name for rankings on openrouter.ai"
            )
            st.session_state.openrouter_site_name = site_name
        
        # Model filtering toggle
        show_tools_only = st.checkbox(
            "🔧 Show only tool-capable models",
            value=st.session_state.show_tools_only,
            help="Filter to show only models that support function/tool calling for MCP integration"
        )
        st.session_state.show_tools_only = show_tools_only
        
        # Model selection dropdown
        models_key = f"{llm_provider}_openrouter_models"
        selected_model_key = f"{llm_provider}_openrouter_model"
        
        if st.session_state.get(models_key):
            model_options = []
            for model in st.session_state[models_key]:
                model_options.append((model["display"], model["id"]))
            
            if model_options:
                selected_model = st.selectbox(
                    f"Select {llm_provider.title()} Model",
                    options=[opt[1] for opt in model_options],
                    format_func=lambda x: next(opt[0] for opt in model_options if opt[1] == x),
                    help=f"Top 5 most popular {llm_provider} models on OpenRouter",
                    key=f"{llm_provider}_model_select"
                )
                st.session_state[selected_model_key] = selected_model
                
                # Show model details
                selected_model_data = next(
                    (m for m in st.session_state[models_key] if m["id"] == selected_model),
                    None
                )
                if selected_model_data:
                    with st.expander("Model Details"):
                        st.write(f"**Description:** {selected_model_data.get('description', 'N/A')}")
                        context_length = selected_model_data.get('context_length', 'Unknown')
                        if isinstance(context_length, (int, float)) and context_length > 0:
                            st.write(f"**Context Length:** {int(context_length):,} tokens")
                        else:
                            st.write(f"**Context Length:** {context_length}")
                        
                        pricing = selected_model_data.get('pricing', {})
                        if pricing:
                            try:
                                prompt_cost = float(pricing.get('prompt', 0)) * 1000000
                                completion_cost = float(pricing.get('completion', 0)) * 1000000
                                st.write(f"**Pricing:** ${prompt_cost:.3f} prompt / ${completion_cost:.3f} completion per 1M tokens")
                            except (ValueError, TypeError):
                                st.write("**Pricing:** Information unavailable")
        else:
            st.info(f"Models will be automatically loaded when switching to {llm_provider.title()}")
    elif llm_provider == "ollama":
        # Ollama Host input
        ollama_host = st.text_input(
            "Ollama Host (Optional)",
            value=st.session_state.ollama_host,
            help="Enter the Ollama server URL (e.g., 'http://localhost:11434'). Leave blank to use default."
        )
        st.session_state.ollama_host = ollama_host
        
        # Model selection
        # If we have models, show a dropdown, otherwise show a text input
        if st.session_state.ollama_models:
            # Add a default option if the current model is not in the list
            model_options = st.session_state.ollama_models.copy()
            if st.session_state.ollama_model not in model_options:
                model_options = [st.session_state.ollama_model] + model_options
            
            ollama_model = st.selectbox(
                "Ollama Model",
                model_options,
                index=model_options.index(st.session_state.ollama_model) if st.session_state.ollama_model in model_options else 0,
                help="Select an Ollama model from the list. Models are automatically loaded when switching to Ollama."
            )
            st.session_state.ollama_model = ollama_model
        else:
            # No models found, show a text input
            ollama_model = st.text_input(
                "Ollama Model Name",
                value=st.session_state.ollama_model,
                help="Enter the name of the locally available Ollama model (e.g., 'llama3', 'mistral'). Models are automatically detected when switching to Ollama."
            )
            st.session_state.ollama_model = ollama_model

    st.markdown("""
    <div style="margin: 1.5rem 0 1rem 0;">
        <h3 style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
            💬 Chat Mode
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat mode selection
    chat_mode = st.selectbox(
        "Mode",
        ["auto", "chat", "tools"],
        index=["auto", "chat", "tools"].index(st.session_state.chat_mode),
        help="Auto: LLM decides when to use tools. Chat: Direct conversation without tools. Tools: Always try to use tools."
    )
    st.session_state.chat_mode = chat_mode
    
    # Show mode description
    if chat_mode == "auto":
        st.caption("🤖 LLM automatically decides when tools are needed")
    elif chat_mode == "chat":
        st.caption("💬 Direct conversation without MCP tools")
    elif chat_mode == "tools":
        st.caption("🔧 Always attempt to use MCP tools")

    st.markdown("""
    <div style="margin: 1.5rem 0 1rem 0;">
        <h3 style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
            🔌 Connection Control
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Compact button layout with minimal spacing
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Connect", help="Connect to the specified server and LLM provider."):
            with st.spinner("Connecting..."):
                connect_to_server()
    with col2:
        if st.button("Disconnect", disabled=not st.session_state.connected):
            disconnect_from_server()
    
    # Compact status display with modern styling
    status_color = "var(--success)" if st.session_state.connected else "var(--error)"
    status_icon = "🟢" if st.session_state.connected else "🔴"
    status_text = "Connected" if st.session_state.connected else "Not Connected"
    
    # Get model info for display
    if st.session_state.connected:
        if st.session_state.llm_provider in ['openai', 'anthropic', 'google']:
            model_raw = st.session_state.get(f'{st.session_state.llm_provider}_openrouter_model')
            model_display = model_raw.split('/')[-1] if model_raw and '/' in model_raw else (model_raw or 'Not selected')
        else:
            model_display = st.session_state.ollama_model
    else:
        model_display = "Not connected"
    
    # Build the status HTML properly
    status_html = f"""
    <div style="background: var(--secondary-bg); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: var(--space-lg); margin: var(--space-md) 0;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--space-sm);">
            <div style="display: flex; align-items: center; gap: var(--space-sm);">
                <span style="font-size: 1.2rem;">{status_icon}</span>
                <span style="font-weight: 600; color: {status_color};">{status_text}</span>
            </div>"""
    
    if st.session_state.connected:
        status_html += f'<div style="color: var(--text-secondary); font-size: 0.8rem;">🎯 {model_display}</div>'
    
    status_html += "</div>"
    
    if st.session_state.connected:
        status_html += f"""
        <div style="color: var(--text-secondary); font-size: 0.875rem; line-height: 1.4; display: grid; grid-template-columns: 1fr; gap: 0.25rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span>📡</span>
                <span style="font-size: 0.8rem; word-break: break-all;">{st.session_state.mcp_endpoint}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span>🤖</span>
                <span>{st.session_state.llm_provider.title()}</span>
            </div>"""
        
        if st.session_state.llm_provider == "ollama" and st.session_state.ollama_host:
            status_html += f"""
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span>🌐</span>
                <span>{st.session_state.ollama_host}</span>
            </div>"""
        
        status_html += "</div>"
    
    status_html += "</div>"
    
    st.markdown(status_html, unsafe_allow_html=True)
    
    # Show connection error if any
    if st.session_state.connection_error:
        st.error(f"Last error: {st.session_state.connection_error}")
    
    # Tools
    if st.session_state.connected and st.session_state.tools:
        st.markdown("""
        <div style="margin: 2rem 0 1rem 0;">
            <h3 style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                🛠️ Available Tools
            </h3>
        </div>
        """, unsafe_allow_html=True)
        for tool in st.session_state.tools:
            with st.expander(tool.name):
                st.write(tool.description)
                if hasattr(tool, 'parameters') and tool.parameters:
                    st.write("**Parameters:**")
                    for param in tool.parameters:
                        required = "Required" if param.required else "Optional"
                        st.write(f"- **{param.name}** ({required}): {param.description}")

# Main content area
st.markdown("---")

if not st.session_state.connected and st.session_state.chat_mode != "chat":
    st.markdown("""
    <div class="welcome-card fade-in">
        <h2>✨ Welcome to MCP Tool Tester</h2>
        <p>Connect to an MCP server to unlock powerful AI tool integration, or switch to chat mode for direct LLM conversation.</p>
        <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid var(--accent-primary); border-radius: 0.5rem; padding: 0.75rem 1rem; color: var(--accent-primary);">
                👈 Configure Connection
            </div>
            <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid var(--accent-secondary); border-radius: 0.5rem; padding: 0.75rem 1rem; color: var(--accent-secondary);">
                💬 Try Chat Mode
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show debug info
    st.subheader("Debug Information")
    st.write(f"Current endpoint: {st.session_state.mcp_endpoint}")
    st.write(f"Current provider: {st.session_state.llm_provider}")
    st.write(f"Current mode: {st.session_state.chat_mode}")
    if st.session_state.llm_provider == "openai":
        st.write(f"OpenAI model: {st.session_state.openai_model}")
    elif st.session_state.llm_provider == "anthropic":
        st.write(f"Anthropic model: {st.session_state.anthropic_model}")
    elif st.session_state.llm_provider == "ollama":
        st.write(f"Ollama model: {st.session_state.ollama_model}")
        st.write(f"Ollama host: {st.session_state.ollama_host or 'default'}")
        st.write(f"Available models: {len(st.session_state.ollama_models)} found")
    if st.session_state.connection_error:
        st.error(f"Connection error: {st.session_state.connection_error}")
else:
    # Chat interface
    if not st.session_state.messages:
        mode_descriptions = {
            "auto": "🤖 The LLM will automatically decide when to use MCP tools based on your questions.",
            "chat": "💬 Direct conversation with the LLM without using MCP tools.",
            "tools": "🔧 The LLM will always attempt to use MCP tools to answer your questions."
        }
        
        mode_icons = {
            "auto": "🤖",
            "chat": "💬",
            "tools": "🔧"
        }
        
        st.markdown(f"""
        <div class="welcome-card fade-in" style="margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">{mode_icons[st.session_state.chat_mode]}</span>
                <h3 class="gradient-text" style="margin: 0;">Ready to Chat!</h3>
            </div>
            <div style="background: var(--tertiary-bg); border: 1px solid var(--accent-primary); border-radius: var(--radius-lg); padding: var(--space-lg); margin-top: var(--space-lg);">
                <div style="color: var(--accent-primary); font-weight: 600; margin-bottom: var(--space-sm);">
                    Current Mode: {st.session_state.chat_mode.title()}
                </div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.5;">
                    {mode_descriptions[st.session_state.chat_mode]}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.write(content)
    
    # Clear conversation button
    if st.session_state.messages:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                result = process_user_message(prompt)
                
                # Display the LLM response
                st.write(result)
                
                # Check if we have enhanced response data from the last processing
                if hasattr(st.session_state, 'last_response_data') and st.session_state.last_response_data:
                    response_data = st.session_state.last_response_data
                    tool_call = response_data.get("tool_call")
                    tool_result = response_data.get("tool_result")
                    has_tools = response_data.get("has_tools", False)
                    processing_steps = response_data.get("processing_steps", [])
                    metadata = response_data.get("metadata", {})
                    
                    # Display tool usage information
                    if tool_call and tool_result:
                        tool_name = tool_call.get('name', 'Unknown')
                        if tool_result.error_code == 0:
                            # Show success message
                            st.success(f"✅ Auto mode: LLM successfully used MCP tool '{tool_name}' and processed the results")
                        else:
                            st.error(f"❌ Auto mode: MCP tool '{tool_name}' failed")
                            st.error(f"**Error:** {tool_result.content}")
                    else:
                        if has_tools:
                            st.info("ℹ️ Auto mode: LLM chose not to use any MCP tools for this query")
                    
                    # Enhanced Raw LLM Response Data Expander
                    with st.expander("🔍 View Raw LLM Response Data", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Initial LLM Response")
                            if response_data.get("raw_initial_response"):
                                # Convert response object to dict for JSON display
                                try:
                                    if hasattr(response_data["raw_initial_response"], '__dict__'):
                                        initial_dict = vars(response_data["raw_initial_response"])
                                    else:
                                        initial_dict = response_data["raw_initial_response"]
                                    st.json(initial_dict)
                                except Exception as e:
                                    st.code(str(response_data["raw_initial_response"]))
                            else:
                                st.info("No initial response data")
                        
                        with col2:
                            st.subheader("Final LLM Response")
                            if response_data.get("raw_final_response"):
                                # Convert response object to dict for JSON display
                                try:
                                    if hasattr(response_data["raw_final_response"], '__dict__'):
                                        final_dict = vars(response_data["raw_final_response"])
                                    else:
                                        final_dict = response_data["raw_final_response"]
                                    st.json(final_dict)
                                except Exception as e:
                                    st.code(str(response_data["raw_final_response"]))
                            else:
                                st.info("No final response data")
                        
                        # Response Metadata
                        if metadata:
                            st.subheader("Response Metadata")
                            st.json(metadata)
                    
                    # Tool Execution Details Expander
                    if tool_call and tool_result:
                        with st.expander("🔧 View Tool Execution Details", expanded=False):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Tool Call")
                                st.json(tool_call)
                            
                            with col2:
                                st.subheader("Tool Result")
                                st.write(f"**Error Code:** {tool_result.error_code}")
                                if tool_result.error_code == 0:
                                    st.success("✅ Tool executed successfully")
                                else:
                                    st.error("❌ Tool execution failed")
                            
                            st.subheader("Tool Output")
                            formatted_result = format_tool_result(tool_result.content)
                            st.code(formatted_result, language="json")
                    
                    # Debug Information Expander
                    with st.expander("🐛 Debug Information", expanded=False):
                        if processing_steps:
                            st.subheader("Processing Steps Timeline")
                            for i, step in enumerate(processing_steps):
                                with st.container():
                                    st.write(f"**Step {i+1}: {step.get('step', 'Unknown').replace('_', ' ').title()}**")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if step.get('timestamp'):
                                            st.write(f"⏰ {step['timestamp']}")
                                    with col2:
                                        if step.get('duration'):
                                            st.write(f"⚡ {step['duration']:.3f}s")
                                    if step.get('data'):
                                        st.write(f"📝 {step['data']}")
                                    st.divider()
                        
                        st.subheader("Session Debug Info")
                        debug_info = {
                            "Provider": metadata.get('provider', 'Unknown'),
                            "Model": metadata.get('model', 'Unknown'),
                            "Base URL": metadata.get('base_url', 'Unknown'),
                            "Has Tools": metadata.get('has_tools', 'Unknown'),
                            "Total Execution Time": f"{metadata.get('execution_time', 0):.3f}s" if metadata.get('execution_time') else 'Unknown'
                        }
                        st.json(debug_info)
                        
                        # Show final LLM content for debugging
                        if response_data.get("final_llm_content"):
                            st.subheader("Final LLM Content (Displayed to User)")
                            st.code(response_data["final_llm_content"])
                    
                    # Clear the response data
                    st.session_state.last_response_data = None
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": result})

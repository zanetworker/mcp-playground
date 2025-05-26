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
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", "")
    }
if "mcp_endpoint" not in st.session_state: 
    st.session_state.mcp_endpoint = "http://localhost:8001/sse"
if "llm_provider" not in st.session_state: 
    st.session_state.llm_provider = "openai"
if "ollama_model" not in st.session_state: 
    st.session_state.ollama_model = DEFAULT_OLLAMA_MODEL
if "ollama_host" not in st.session_state: 
    st.session_state.ollama_host = ""
if "ollama_models" not in st.session_state: 
    st.session_state.ollama_models = []
if "chat_mode" not in st.session_state: 
    st.session_state.chat_mode = "auto"  # auto, chat, tools

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
    
    elif st.session_state.llm_provider == "openai":
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=st.session_state.api_keys["openai"])
            
            response = await client.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=[{"role": "user", "content": user_input}]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error chatting with OpenAI: {e}"
    
    elif st.session_state.llm_provider == "anthropic":
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=st.session_state.api_keys["anthropic"])
            
            response = await client.messages.create(
                model=DEFAULT_ANTHROPIC_MODEL,
                max_tokens=1000,
                messages=[{"role": "user", "content": user_input}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error chatting with Anthropic: {e}"
    
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
        if st.session_state.llm_provider == "openai" and st.session_state.api_keys["openai"]:
            llm_bridge = OpenAIBridge(client, api_key=st.session_state.api_keys["openai"])
        elif st.session_state.llm_provider == "anthropic" and st.session_state.api_keys["anthropic"]:
            llm_bridge = AnthropicBridge(client, api_key=st.session_state.api_keys["anthropic"])
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
def extract_content_from_llm_response(llm_response):
    """Extract clean text content from different LLM provider response formats.
    
    Args:
        llm_response: Response object from OpenAI, Anthropic, Ollama, or dict
        
    Returns:
        str: Clean text content extracted from the response (never None)
    """
    try:
        # OpenAI ChatCompletion object
        if hasattr(llm_response, 'choices') and llm_response.choices:
            content = llm_response.choices[0].message.content
            return content if content is not None else "No content received from OpenAI"
        
        # Anthropic Message object
        elif hasattr(llm_response, 'content') and llm_response.content:
            for content in llm_response.content:
                if hasattr(content, 'type') and content.type == "text":
                    text = content.text
                    return text if text is not None else "No text content from Anthropic"
            # Fallback: return first content item as string
            first_content = str(llm_response.content[0])
            return first_content if first_content else "Empty content from Anthropic"
        
        # Ollama dict response
        elif isinstance(llm_response, dict):
            if 'message' in llm_response:
                message = llm_response['message']
                if isinstance(message, dict) and 'content' in message:
                    content = message['content']
                    return content if content is not None else "No content in Ollama message"
            # Fallback for other dict formats
            content = llm_response.get('content', str(llm_response))
            return content if content else "Empty Ollama response"
        
        # String response (already clean)
        elif isinstance(llm_response, str):
            return llm_response if llm_response else "Empty string response"
        
        # None response
        elif llm_response is None:
            return "No response received"
        
        # Fallback: convert to string
        fallback = str(llm_response)
        return fallback if fallback else "Empty response object"
        
    except Exception as e:
        return f"Error extracting content: {e}"

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
            result = await st.session_state.llm_bridge.process_query(user_input)
            
            # Format the response nicely
            if isinstance(result, dict):
                llm_response = result.get("llm_response", {})
                tool_call = result.get("tool_call")
                tool_result = result.get("tool_result")
                
                # Extract the main message content using unified parser
                content = extract_content_from_llm_response(llm_response)
                
                response_parts = []
                
                # Only add content if it's not a generic "no content" message and we have a tool result
                if tool_call and tool_result:
                    # If we have a tool result, prioritize that over generic "no content" messages
                    if not content.startswith("No content received from"):
                        response_parts.append(content)
                    
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
                    response_parts.append(content)
                
                return "\n".join(response_parts)
            else:
                return extract_content_from_llm_response(result)
                
        except Exception as e:
            return f"Sorry, I encountered an error: {e}"
    
    # Auto mode: let LLM decide whether to use tools
    else:  # auto mode
        if not st.session_state.llm_bridge:
            # Fall back to direct chat if no bridge available
            return await chat_with_llm_directly(user_input)
        
        try:
            result = await st.session_state.llm_bridge.process_query(user_input)
            
            # Format the response nicely
            if isinstance(result, dict):
                llm_response = result.get("llm_response", {})
                tool_call = result.get("tool_call")
                tool_result = result.get("tool_result")
                
                # Extract the main message content using unified parser
                content = extract_content_from_llm_response(llm_response)
                
                response_parts = []
                
                # Only add content if it's not a generic "no content" message and we have a tool result
                if tool_call and tool_result:
                    # If we have a tool result, prioritize that over generic "no content" messages
                    if not content.startswith("No content received from"):
                        response_parts.append(content)
                    
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
                    response_parts.append(content)
                
                return "\n".join(response_parts)
            else:
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

# Title
st.title("MCP Tool Tester")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Server Connection")
    
    # MCP Endpoint
    mcp_endpoint = st.text_input(
        "MCP Endpoint URL",
        value=st.session_state.mcp_endpoint,
        help="Enter the full URL for the MCP SSE server."
    )
    st.session_state.mcp_endpoint = mcp_endpoint

    st.subheader("LLM Configuration")
    
    # LLM Provider
    llm_provider = st.selectbox(
        "LLM Provider",
        ["openai", "anthropic", "ollama"],
        index=["openai", "anthropic", "ollama"].index(st.session_state.llm_provider)
    )
    st.session_state.llm_provider = llm_provider
    
    # Provider specific settings
    if llm_provider == "openai":
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.api_keys["openai"]
        )
        st.session_state.api_keys["openai"] = api_key
    elif llm_provider == "anthropic":
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.api_keys["anthropic"]
        )
        st.session_state.api_keys["anthropic"] = api_key
    elif llm_provider == "ollama":
        # Ollama Host input
        ollama_host = st.text_input(
            "Ollama Host (Optional)",
            value=st.session_state.ollama_host,
            help="Enter the Ollama server URL (e.g., 'http://localhost:11434'). Leave blank to use default."
        )
        st.session_state.ollama_host = ollama_host
        
        # Refresh Ollama models button
        if st.button("Refresh Ollama Models", key="refresh_ollama_models"):
            with st.spinner("Fetching Ollama models..."):
                try:
                    # Fetch models with proper event loop handling
                    try:
                        loop = asyncio.get_running_loop()
                        # If we're in an existing loop, use thread executor
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, fetch_ollama_models(st.session_state.ollama_host))
                            models = future.result(timeout=30)
                    except RuntimeError:
                        # No event loop running, safe to use asyncio.run()
                        models = asyncio.run(fetch_ollama_models(st.session_state.ollama_host))
                    
                    if models:
                        # Ensure models are stored as a list of strings
                        st.session_state.ollama_models = [str(model) for model in models]
                        st.success(f"Found {len(models)} Ollama models: {', '.join(st.session_state.ollama_models)}")
                        # Force a rerun to update the UI
                        st.rerun()
                    else:
                        st.warning("No Ollama models found. Is Ollama running?")
                except Exception as e:
                    st.error(f"Error fetching Ollama models: {e}")
        
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
                help="Select an Ollama model from the list. Click 'Refresh Ollama Models' to update the list."
            )
            st.session_state.ollama_model = ollama_model
        else:
            # No models found, show a text input
            ollama_model = st.text_input(
                "Ollama Model Name",
                value=st.session_state.ollama_model,
                help="Enter the name of the locally available Ollama model (e.g., 'llama3', 'mistral'). Click 'Refresh Ollama Models' to detect available models."
            )
            st.session_state.ollama_model = ollama_model

    st.subheader("Chat Mode")
    
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

    st.subheader("Connection Control")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Connect", help="Connect to the specified server and LLM provider."):
            with st.spinner("Connecting..."):
                connect_to_server()
    with col2:
        if st.button("Disconnect", disabled=not st.session_state.connected):
            disconnect_from_server()
    
    # Status
    st.markdown(
        f"**Status:** {'🟢 Connected' if st.session_state.connected else '🔴 Not connected'}"
    )
    if st.session_state.connected:
        st.caption(f"Server: {st.session_state.mcp_endpoint}")
        st.caption(f"LLM: {st.session_state.llm_provider}")
        if st.session_state.llm_provider == "ollama":
            st.caption(f"Model: {st.session_state.ollama_model}")
            if st.session_state.ollama_host:
                st.caption(f"Host: {st.session_state.ollama_host}")
    
    # Show connection error if any
    if st.session_state.connection_error:
        st.error(f"Last error: {st.session_state.connection_error}")
    
    # Tools
    if st.session_state.connected and st.session_state.tools:
        st.header("Available Tools")
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
    <div style="text-align: center; padding: 2rem; background-color: #1e1e2e; border-radius: 10px; margin: 2rem 0;">
        <h2 style="color: #ffffff; font-weight: bold;">Welcome to MCP Tool Tester</h2>
        <p style="font-size: 1.2rem; color: #ffffff;">Please connect to an MCP server using the sidebar to get started.</p>
        <p style="color: #ffffff;">👈 Configure your connection settings in the sidebar</p>
        <p style="color: #ffffff;">Or switch to "chat" mode to talk directly with the LLM</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show debug info
    st.subheader("Debug Information")
    st.write(f"Current endpoint: {st.session_state.mcp_endpoint}")
    st.write(f"Current provider: {st.session_state.llm_provider}")
    st.write(f"Current mode: {st.session_state.chat_mode}")
    if st.session_state.llm_provider == "ollama":
        st.write(f"Ollama model: {st.session_state.ollama_model}")
        st.write(f"Ollama host: {st.session_state.ollama_host or 'default'}")
        st.write(f"Available models: {len(st.session_state.ollama_models)} found")
    if st.session_state.connection_error:
        st.error(f"Connection error: {st.session_state.connection_error}")
else:
    # Chat interface
    if not st.session_state.messages:
        mode_descriptions = {
            "auto": "The LLM will automatically decide when to use MCP tools based on your questions.",
            "chat": "Direct conversation with the LLM without using MCP tools.",
            "tools": "The LLM will always attempt to use MCP tools to answer your questions."
        }
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #1e1e2e; border-radius: 10px; margin-bottom: 1rem;">
            <h3 style="color: #ffffff; font-weight: bold;">Ready to chat!</h3>
            <p style="color: #ffffff;">Current mode: <strong>{st.session_state.chat_mode.title()}</strong></p>
            <p style="color: #ffffff; font-size: 0.9rem;">{mode_descriptions[st.session_state.chat_mode]}</p>
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
                st.write(result)
                # Add assistant response to chat
                st.session_state.messages.append({"role": "assistant", "content": result})

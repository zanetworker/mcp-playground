import streamlit as st
import sys
import os
import json
import asyncio
import datetime
import markdown  # Import markdown library
import ollama  # Import ollama for model listing
from typing import List, Dict, Any, Optional, Union

# Add parent directory to path to import mcp_sse_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the MCP SSE client and model definitions
from mcp_sse_client.client import MCPClient
from mcp_sse_client.llm_bridge.openai_bridge import OpenAIBridge
from mcp_sse_client.llm_bridge.anthropic_bridge import AnthropicBridge
from mcp_sse_client.llm_bridge.ollama_bridge import OllamaBridge # Import OllamaBridge
from mcp_sse_client.llm_bridge.models import (
    OPENAI_MODELS, DEFAULT_OPENAI_MODEL,
    ANTHROPIC_MODELS, DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OLLAMA_MODEL # Import Ollama default
)

# Set page config
st.set_page_config(
    page_title="MCP Tool Tester",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* General layout */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Buttons */
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
        margin-bottom: 2px; /* Reduced space between buttons */
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    
    /* Add spacing between Connect/Disconnect buttons in columns */
    div[data-testid="stHorizontalBlock"] > div:first-child .stButton button {
        margin-right: 10px; /* Add space to the right of the Connect button */
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #2b2b2b !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
        border-radius: 4px !important;
    }
    .stTextArea textarea {
        font-family: monospace; /* Use monospace font for text areas */
    }
    
    /* Chat input */
    .stChatInput input {
        background-color: #2b2b2b;
        color: #ffffff;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 10px;
    }
    
    /* Tool styling */
    .tool-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .tool-description {
        margin-bottom: 1rem;
    }
    .param-required {
        color: #ff4b4b;
        font-weight: bold;
    }
    .param-optional {
        color: #4b8bff;
    }
    .param-name {
        font-weight: bold;
    }
    .param-type {
        color: #888;
        font-style: italic;
    }
    
    /* Message boxes */
    .reasoning-box, .tool-call-box, .tool-result-box {
        background-color: #2d3748;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .reasoning-box { border-left: 4px solid #4299e1; }
    .tool-call-box { border-left: 4px solid #f6ad55; }
    .tool-result-box { border-left: 4px solid #68d391; }
    
    /* Code blocks */
    .stCodeBlock, code {
        background-color: #1e1e1e !important;
        border-radius: 4px !important;
        max-height: 500px;
        overflow-y: auto;
        color: #d4d4d4; /* Light text color for code */
    }
    
    /* Make code wrap properly */
    pre, code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }
    
    /* Ensure tool result content is properly formatted */
    .tool-result-box pre {
        max-width: 100%;
        overflow-x: hidden;
    }
    
    /* Increase the width of the main content area */
    .main .block-container {
        max-width: 1200px;
        padding-left: 5%;
        padding-right: 5%;
    }
    
    /* Sidebar History */
    .sidebar .stExpander {
        border: 1px solid #444;
        border-radius: 4px;
        margin-bottom: 5px;
    }
    .sidebar .stExpander header {
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state (Ensure keys used in callbacks/buttons exist)
if "connected" not in st.session_state: st.session_state.connected = False
if "client" not in st.session_state: st.session_state.client = None
if "llm_bridge" not in st.session_state: st.session_state.llm_bridge = None
if "tools" not in st.session_state: st.session_state.tools = []
if "messages" not in st.session_state: st.session_state.messages = []
if "conversation_history" not in st.session_state: st.session_state.conversation_history = []
if "current_conversation_id" not in st.session_state: st.session_state.current_conversation_id = 0
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", "")
    }
if "mcp_endpoint" not in st.session_state: st.session_state.mcp_endpoint = "http://localhost:8000/sse"
if "llm_provider" not in st.session_state: st.session_state.llm_provider = "openai"
if "openai_model" not in st.session_state: st.session_state.openai_model = DEFAULT_OPENAI_MODEL
if "anthropic_model" not in st.session_state: st.session_state.anthropic_model = DEFAULT_ANTHROPIC_MODEL
if "ollama_model" not in st.session_state: st.session_state.ollama_model = DEFAULT_OLLAMA_MODEL
if "ollama_host" not in st.session_state: st.session_state.ollama_host = ""
if "ollama_models" not in st.session_state: st.session_state.ollama_models = []
if "server_history" not in st.session_state: st.session_state.server_history = []
if "last_interaction_time" not in st.session_state: st.session_state.last_interaction_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Ensure widget keys exist if they might be accessed before creation
if "llm_provider_select" not in st.session_state: st.session_state.llm_provider_select = st.session_state.llm_provider
if "openai_api_key_input" not in st.session_state: st.session_state.openai_api_key_input = st.session_state.api_keys.get("openai", "")
if "openai_model_select" not in st.session_state: st.session_state.openai_model_select = st.session_state.openai_model
if "anthropic_api_key_input" not in st.session_state: st.session_state.anthropic_api_key_input = st.session_state.api_keys.get("anthropic", "")
if "anthropic_model_select" not in st.session_state: st.session_state.anthropic_model_select = st.session_state.anthropic_model
if "ollama_model_input" not in st.session_state: st.session_state.ollama_model_input = st.session_state.ollama_model
if "ollama_host_input" not in st.session_state: st.session_state.ollama_host_input = st.session_state.ollama_host


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
        
        # Debug the response structure
        print(f"Ollama models response: {models_info}")
        
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
        
        # Log the extracted model names
        print(f"Extracted model names: {model_names}")
        
        return model_names
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []

# --- Callback Functions ---
def update_llm_settings_state():
    """Updates the session state for the selected LLM provider.
       This callback is triggered when the provider dropdown changes.
       The actual model settings are handled when the Connect button is pressed.
    """
    # Only update the provider selection - this is safe because the provider dropdown
    # widget is already created and its value is available in session state
    provider = st.session_state.llm_provider_select
    
    # Update the main provider state if it changed
    if st.session_state.llm_provider != provider:
        st.session_state.llm_provider = provider
        
        # Notify user if already connected
        if st.session_state.connected:
            st.session_state.messages.append({
                "role": "system", 
                "content": f"LLM provider changed to {provider}. Press Connect to apply changes."
            })

# --- Async Helper ---
async def fetch_tools_async(client):
    """Asynchronously fetches tools from the MCP client."""
    if not client:
        return []
    try:
        return await client.list_tools()
    except Exception as e:
        st.error(f"Error fetching tools: {e}")
        return []

# Function to save current conversation to history
def save_current_conversation():
    if st.session_state.messages and len(st.session_state.messages) > 0:
        timestamp = st.session_state.get("last_interaction_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        conversation = {
            "id": st.session_state.current_conversation_id,
            "timestamp": timestamp,
            "server": st.session_state.mcp_endpoint,
            "messages": st.session_state.messages.copy(),
            "summary": get_conversation_summary(st.session_state.messages)
        }
        
        # Check if we're updating an existing conversation
        existing_index = next((i for i, conv in enumerate(st.session_state.conversation_history) if conv["id"] == st.session_state.current_conversation_id), None)
        if existing_index is not None:
            st.session_state.conversation_history[existing_index] = conversation
        else:
            # Otherwise add as new conversation
            st.session_state.conversation_history.append(conversation)
            st.session_state.current_conversation_id = len(st.session_state.conversation_history) # Assign new ID for next potential conversation

# Function to get a short summary of the conversation
def get_conversation_summary(messages):
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    if not user_messages:
        return "Empty conversation"
    first_msg = user_messages[0]["content"]
    return first_msg[:60] + "..." if len(first_msg) > 60 else first_msg

# Function to reset the current conversation
def reset_conversation():
    save_current_conversation()  # Save current conversation before resetting
    st.session_state.messages = []
    st.session_state.current_conversation_id = len(st.session_state.conversation_history) # Prepare for a new conversation
    st.session_state.last_interaction_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Function to load a conversation from history
def load_conversation(conversation_id):
    for conv in st.session_state.conversation_history:
        if conv["id"] == conversation_id:
            save_current_conversation() # Save the current one first
            st.session_state.messages = conv["messages"].copy()
            st.session_state.current_conversation_id = conversation_id
            loaded_server = conv["server"]
            current_server = st.session_state.mcp_endpoint
            
            st.session_state.mcp_endpoint = loaded_server # Update endpoint in session state
            st.session_state.last_interaction_time = conv["timestamp"]

            # Only disconnect if the server has changed or was not connected
            if not st.session_state.connected or current_server != loaded_server:
                st.session_state.connected = False 
                st.session_state.client = None
                st.session_state.llm_bridge = None
                st.session_state.tools = []
            # If the server is the same, keep the connection, client, bridge, tools
            return True
    return False

# Title
st.title("MCP Tool Tester")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Server Configuration Tab and History Tab
    server_tab, history_tab = st.tabs(["Server Config", "History"])
    
    with server_tab:
        st.subheader("Server Connection")
        
        # Use only text input for MCP Endpoint
        mcp_endpoint_widget = st.text_input(
            "MCP Endpoint URL",
            value=st.session_state.mcp_endpoint,
            key="mcp_endpoint_input",
            help="Enter the full URL for the MCP SSE server. Previously used servers will be remembered."
        )

        st.subheader("LLM Configuration")
        # LLM Provider
        llm_provider_options = ["openai", "anthropic", "ollama"] # Add ollama
        llm_provider_widget = st.selectbox(
            "LLM Provider",
            llm_provider_options,
            index=llm_provider_options.index(st.session_state.llm_provider) if st.session_state.llm_provider in llm_provider_options else 0,
            key="llm_provider_select",
            on_change=update_llm_settings_state 
        )
        
        # --- Provider Specific Settings ---
        # Use local variables for widget creation, state updated via callback or connect button
        current_provider = st.session_state.llm_provider_select # Read current selection

        if current_provider == "openai":
            openai_model_widget = st.selectbox(
                "OpenAI Model",
                OPENAI_MODELS, 
                index=OPENAI_MODELS.index(st.session_state.openai_model) if st.session_state.openai_model in OPENAI_MODELS else 0,
                key="openai_model_select", 
                on_change=update_llm_settings_state, 
                help="Select the OpenAI model to use."
            )
            openai_api_key_widget = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.api_keys["openai"],
                key="openai_api_key_input"
            )
        elif current_provider == "anthropic":
            anthropic_model_widget = st.selectbox(
                "Anthropic Model",
                ANTHROPIC_MODELS, 
                index=ANTHROPIC_MODELS.index(st.session_state.anthropic_model) if st.session_state.anthropic_model in ANTHROPIC_MODELS else 0,
                key="anthropic_model_select", 
                on_change=update_llm_settings_state, 
                help="Select the Anthropic model to use."
            )
            anthropic_api_key_widget = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.api_keys["anthropic"],
                key="anthropic_api_key_input"
            )
        elif current_provider == "ollama":
            # Ollama Host input
            ollama_host_widget = st.text_input(
                "Ollama Host (Optional)",
                value=st.session_state.ollama_host,
                key="ollama_host_input", 
                help="Enter the Ollama server URL (e.g., 'http://localhost:11434'). Leave blank to use default."
            )
            
            # Refresh Ollama models button
            if st.button("Refresh Ollama Models", key="refresh_ollama_models"):
                with st.spinner("Fetching Ollama models..."):
                    try:
                        # Update the host in session state
                        st.session_state.ollama_host = st.session_state.ollama_host_input
                        # Fetch models
                        models = asyncio.run(fetch_ollama_models(st.session_state.ollama_host))
                        
                        # Debug output
                        print(f"Fetched Ollama models: {models}")
                        
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
                        print(f"Exception details: {type(e).__name__}: {str(e)}")
            
            # Model selection
            # If we have models, show a dropdown, otherwise show a text input
            if st.session_state.ollama_models:
                # Add a default option if the current model is not in the list
                model_options = st.session_state.ollama_models.copy()
                if st.session_state.ollama_model not in model_options:
                    model_options = [st.session_state.ollama_model] + model_options
                
                ollama_model_widget = st.selectbox(
                    "Ollama Model",
                    model_options,
                    index=model_options.index(st.session_state.ollama_model) if st.session_state.ollama_model in model_options else 0,
                    key="ollama_model_input", 
                    on_change=update_llm_settings_state, 
                    help="Select an Ollama model from the list. Click 'Refresh Ollama Models' to update the list."
                )
            else:
                # No models found, show a text input
                ollama_model_widget = st.text_input(
                    "Ollama Model Name",
                    value=st.session_state.ollama_model,
                    key="ollama_model_input", 
                    on_change=update_llm_settings_state, 
                    help="Enter the name of the locally available Ollama model (e.g., 'llama3', 'mistral'). Click 'Refresh Ollama Models' to detect available models."
                )

        st.subheader("Connection Control")
        col1, col2 = st.columns(2)
        with col1:
            # Connect Button - Shortened Text
            connect_button = st.button(
                "Connect",
                key="connect_button",
                help="Connect to the specified server and LLM provider."
            )
        with col2:
            # Disconnect Button
            disconnect_button = st.button(
                "Disconnect",
                key="disconnect_button",
                disabled=not st.session_state.connected
            )
        
        # Reset Conversation Button
        if st.session_state.connected:
            if st.button("New Conversation", key="reset_conversation_button", help="Save current chat and start a new one."):
                reset_conversation()
                st.rerun()
        
        # Status
        st.markdown(
            f"**Status:** {'🟢 Connected' if st.session_state.connected else '🔴 Not connected'}",
            unsafe_allow_html=True
        )
        if st.session_state.connected:
             st.caption(f"Server: {st.session_state.mcp_endpoint}")
             # Display selected model based on provider
             provider_display = st.session_state.llm_provider
             if provider_display == "openai":
                 model_display = st.session_state.openai_model
                 st.caption(f"LLM: OpenAI ({model_display})")
             elif provider_display == "anthropic":
                 model_display = st.session_state.anthropic_model
                 st.caption(f"LLM: Anthropic ({model_display})")
             elif provider_display == "ollama":
                 model_display = st.session_state.ollama_model
                 host_display = f", Host: {st.session_state.ollama_host}" if st.session_state.ollama_host else ""
                 st.caption(f"LLM: Ollama ({model_display}{host_display})")


    with history_tab:
        st.subheader("Conversation History")
        
        if not st.session_state.conversation_history:
            st.info("No conversation history yet.")
        else:
            # Display conversation history (newest first)
            for i, conv in enumerate(reversed(st.session_state.conversation_history)):
                conv_index = len(st.session_state.conversation_history) - 1 - i
                with st.expander(f"{conv['timestamp']} - {conv['summary']}"):
                    st.caption(f"Server: {conv['server']}")
                    st.caption(f"Messages: {len(conv['messages'])}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Load conversation button
                        if st.button("Load", key=f"load_conv_{conv['id']}"):
                            load_conversation(conv['id'])
                            st.rerun()
                    with col2:
                        # Delete conversation button
                        if st.button("Delete", key=f"delete_conv_{conv['id']}"):
                            st.session_state.conversation_history.pop(conv_index)
                            if st.session_state.current_conversation_id == conv["id"]:
                                reset_conversation() # Start a new one if deleting the current
                            st.rerun()
    
    # Tools (Displayed below tabs)
    if st.session_state.connected and st.session_state.tools:
        st.header("Available Tools")
        
        for tool in st.session_state.tools:
            with st.expander(tool.name):
                st.markdown(f"<div class='tool-description'>{tool.description}</div>", unsafe_allow_html=True)
                
                # Required Parameters
                required_params = [p for p in tool.parameters if p.required]
                if required_params:
                    st.markdown("<div class='param-required'>Required Parameters:</div>", unsafe_allow_html=True)
                    for param in required_params:
                        st.markdown(f"<div class='param-item'><span class='param-name'>{param.name}</span>: <span class='param-type'>({param.parameter_type})</span> {param.description}</div>", unsafe_allow_html=True)
                
                # Optional Parameters
                optional_params = [p for p in tool.parameters if not p.required]
                if optional_params:
                    st.markdown("<div class='param-optional'>Optional Parameters:</div>", unsafe_allow_html=True)
                    for param in optional_params:
                        default_text = f", default: {param.default}" if param.default is not None else ""
                        st.markdown(f"<div class='param-item'><span class='param-name'>{param.name}</span>: <span class='param-type'>({param.parameter_type}{default_text})</span> {param.description}</div>", unsafe_allow_html=True)

# Main content area - Use columns to create a better layout
main_container = st.container()
with main_container:
    # Create a clear visual separation
    st.markdown("---")
    
    if not st.session_state.connected:
        # Show a more prominent message when not connected
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: #1e1e2e; border-radius: 10px; margin: 2rem 0;">
            <h2 style="color: #ffffff; font-weight: bold;">Welcome to MCP Tool Tester</h2>
            <p style="font-size: 1.2rem; color: #ffffff;">Please connect to an MCP server using the sidebar to get started.</p>
            <p style="color: #ffffff;">👈 Configure your connection settings in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Add a container for the chat area with better styling
        chat_container = st.container()
        with chat_container:
            # Add a welcome message if no messages yet
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align: center; padding: 1rem; background-color: #1e1e2e; border-radius: 10px; margin-bottom: 1rem;">
                    <h3 style="color: #ffffff; font-weight: bold;">Ready to chat!</h3>
                    <p style="color: #ffffff;">Type a message below to start interacting with the MCP tools.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Chat messages
            for message_index, message in enumerate(st.session_state.messages): # Added enumerate for index
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    with st.chat_message("user"):
                        st.write(content)
                elif role == "assistant":
                    with st.chat_message("assistant"):
                        st.write(content)
                elif role == "reasoning":
                    with st.chat_message("assistant", avatar="🧠"):
                        st.markdown("<div class='reasoning-box'>", unsafe_allow_html=True)
                        st.markdown(content)
                        st.markdown("</div>", unsafe_allow_html=True)
                elif role == "tool_call":
                    with st.chat_message("assistant", avatar="🔧"):
                        st.markdown("<div class='tool-call-box'>", unsafe_allow_html=True)
                        st.markdown(f"**Selected Tool:** `{content['name']}`")
                        if content["parameters"]:
                            st.markdown("**Parameters:**")
                            for key, value in content["parameters"].items():
                                st.markdown(f"- **{key}:** `{value}`")
                        else:
                            st.markdown("*No parameters provided*")
                        st.markdown("</div>", unsafe_allow_html=True)
                elif role == "tool_result":
                    with st.chat_message("assistant", avatar="📊"):
                        st.markdown("<div class='tool-result-box'>", unsafe_allow_html=True)
                        
                        # Handle both dictionary and object access for backward compatibility
                        if isinstance(content, dict):
                            success = content.get("error_code", 1) == 0
                            content_text = content.get("content", str(content))
                        else: # Assume it's the ToolInvocationResult object
                            success = content.error_code == 0 if hasattr(content, 'error_code') else False
                            content_text = content.content if hasattr(content, 'content') else str(content)
                        
                        st.markdown(f"**Status:** {'✅ Success' if success else '❌ Error'}")
                        
                        # Try to parse as JSON for better formatting
                        try:
                            if isinstance(content_text, str):
                                content_json = json.loads(content_text)
                                
                                # Add tabs for different views
                                json_tab, text_tab = st.tabs(["JSON View", "Text View"])
                                
                                with json_tab:
                                    st.json(content_json)
                                
                                with text_tab:
                                    # Extract text content if available
                                    extracted_text = None
                                    if isinstance(content_json, dict) and "text" in content_json:
                                        extracted_text = content_json["text"]
                                    elif isinstance(content_json, list) and len(content_json) > 0 and isinstance(content_json[0], dict) and "text" in content_json[0]:
                                        extracted_text = "\n\n".join([item.get("text", "") for item in content_json if isinstance(item, dict)])
                                    elif isinstance(content_json, dict):
                                        for field in ["content", "data", "result", "response"]:
                                            if field in content_json and isinstance(content_json[field], dict) and "text" in content_json[field]:
                                                extracted_text = content_json[field]["text"]
                                                break
                                    
                                    if extracted_text:
                                        # Try to detect the format of the text
                                        format_type = "text"
                                        stripped_text = extracted_text.strip()
                                        if stripped_text.startswith('{') and stripped_text.endswith('}'):
                                            try: json.loads(extracted_text); format_type = "json"
                                            except: pass
                                        elif stripped_text.startswith('|') or stripped_text.startswith('#'): format_type = "markdown"
                                        elif stripped_text.startswith('<') and stripped_text.endswith('>'): format_type = "html"
                                        
                                        # Display based on format
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            max_lines = min(15, extracted_text.count('\n') + 1)
                                            height = max(100, min(300, max_lines * 20))
                                            
                                            if format_type == "json": st.json(json.loads(extracted_text))
                                            elif format_type == "markdown": st.markdown(extracted_text)
                                            elif format_type == "html": st.code(extracted_text, language="html")
                                            # Always provide a text area for copying, use message_index for unique key
                                            st.text_area("Copy Text", extracted_text, height=height, key=f"copy_text_{message_index}")
                                        
                                        with col2:
                                            # Add unique key using message index and type
                                            download_key = f"download_text_{message_index}"
                                            st.download_button(
                                                "Download",
                                                extracted_text,
                                                file_name=f"extracted.{format_type if format_type != 'text' else 'txt'}",
                                                mime=f"text/{'plain' if format_type == 'text' else format_type}",
                                                key=download_key
                                            )
                                    else:
                                        # If no text field found, convert the entire JSON to a string
                                        compact_json = json.dumps(content_json, indent=2)
                                        max_lines = min(15, compact_json.count('\n') + 1)
                                        height = max(100, min(300, max_lines * 20))
                                        col1, col2 = st.columns([3, 1])
                                        with col1: st.text_area("Copy Raw JSON", compact_json, height=height, key=f"copy_raw_{message_index}")
                                        with col2: 
                                            # Add unique key using message index and type
                                            download_key = f"download_json_{message_index}"
                                            st.download_button(
                                                "Download JSON", 
                                                compact_json, 
                                                file_name="raw_content.json", 
                                                mime="application/json",
                                                key=download_key
                                            )
                            else: # If content_text is not a string (e.g., already an object)
                                st.json(content_text) # Display as JSON
                        except Exception as display_ex:
                            st.warning(f"Could not parse/display content: {display_ex}")
                            st.code(str(content_text)) # Fallback to raw code display
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                elif role == "system":
                    st.info(content)
            
            # Add a visual separator before the chat input
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            
            # Chat input - Make it more prominent
            st.markdown("<div style='background-color: #1e1e2e; padding: 1rem; border-radius: 10px;'>", unsafe_allow_html=True)
            query = st.chat_input("Type your query here...")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if query:
                # Update last interaction time
                st.session_state.last_interaction_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Add user message
                st.session_state.messages.append({"role": "user", "content": query})
                
                with st.chat_message("user"): st.write(query)
                
                # Process query
                with st.spinner("Processing query..."):
                    try:
                        # Check if bridge exists, if not, maybe connection failed silently
                        if not st.session_state.llm_bridge:
                             st.error("LLM Bridge not initialized. Please connect first.")
                             st.stop() # Stop execution for this query

                        llm_result = asyncio.run(st.session_state.llm_bridge.process_query(query))
                        
                        llm_response = llm_result.get("llm_response")
                        tool_call = llm_result.get("tool_call")
                        tool_result = llm_result.get("tool_result")
                        
                        reasoning, final_response = None, None
                        if llm_response:
                            if hasattr(llm_response, 'choices') and llm_response.choices: # OpenAI
                                message = llm_response.choices[0].message # OpenAI format
                                if message.content:
                                    if tool_call: reasoning = message.content
                                    else: final_response = message.content
                            elif isinstance(llm_response, dict) and 'message' in llm_response: # Ollama format
                                message = llm_response['message']
                                if message.get('content'):
                                    if tool_call: reasoning = message['content']
                                    else: final_response = message['content']
                            elif hasattr(llm_response, 'content'): # Anthropic format
                                text_parts = [c.text for c in llm_response.content if hasattr(c, 'type') and c.type == "text"]
                                content_text = "\n".join(text_parts)
                                if content_text:
                                    if tool_call: reasoning = content_text
                                    else: final_response = content_text
                        
                        if reasoning:
                            st.session_state.messages.append({"role": "reasoning", "content": reasoning})
                            with st.chat_message("assistant", avatar="🧠"):
                                st.markdown("<div class='reasoning-box'>", unsafe_allow_html=True)
                                st.markdown(reasoning)
                                st.markdown("</div>", unsafe_allow_html=True)
                        
                        if tool_call:
                            st.session_state.messages.append({"role": "tool_call", "content": tool_call})
                            with st.chat_message("assistant", avatar="🔧"):
                                st.markdown("<div class='tool-call-box'>", unsafe_allow_html=True)
                                st.markdown(f"**Selected Tool:** `{tool_call['name']}`")
                                if tool_call["parameters"]:
                                    st.markdown("**Parameters:**")
                                    for key, value in tool_call["parameters"].items(): st.markdown(f"- **{key}:** `{value}`")
                                else: st.markdown("*No parameters provided*")
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            st.session_state.messages.append({"role": "tool_result", "content": tool_result})
                            # Display logic for tool_result is handled by the message loop above after rerun
                        
                        if final_response:
                            st.session_state.messages.append({"role": "assistant", "content": final_response})
                            with st.chat_message("assistant"): st.write(final_response)
                        
                        # Save conversation to history and rerun to display new messages
                        save_current_conversation()
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        st.session_state.messages.append({"role": "system", "content": f"Error: {str(e)}"})
                        save_current_conversation() # Save even if there's an error
                        st.rerun()

# Handle connect button
if connect_button:
    with st.spinner("Connecting to MCP server..."):
        try:
            # --- Get configuration values from widget states ---
            endpoint_to_connect = st.session_state.mcp_endpoint_input
            provider_to_connect = st.session_state.llm_provider_select
            
            openai_key_to_use = st.session_state.openai_api_key_input
            openai_model_to_use = st.session_state.openai_model_select
            
            anthropic_key_to_use = st.session_state.anthropic_api_key_input
            anthropic_model_to_use = st.session_state.anthropic_model_select
            
            ollama_model_to_use = st.session_state.ollama_model_input
            ollama_host_to_use = st.session_state.ollama_host_input

            # --- Check if connection needs reset ---
            if not st.session_state.connected or \
               st.session_state.mcp_endpoint != endpoint_to_connect or \
               st.session_state.llm_provider != provider_to_connect:
                save_current_conversation()
                reset_conversation() 

            # --- Update core session state ---
            st.session_state.mcp_endpoint = endpoint_to_connect
            st.session_state.llm_provider = provider_to_connect 

            if provider_to_connect == "openai":
                st.session_state.api_keys["openai"] = openai_key_to_use
                st.session_state.openai_model = openai_model_to_use
            elif provider_to_connect == "anthropic":
                st.session_state.api_keys["anthropic"] = anthropic_key_to_use
                st.session_state.anthropic_model = anthropic_model_to_use
            elif provider_to_connect == "ollama":
                st.session_state.ollama_model = ollama_model_to_use
                st.session_state.ollama_host = ollama_host_to_use

            # Add endpoint to history
            if endpoint_to_connect and endpoint_to_connect.startswith(("http://", "https://")) and endpoint_to_connect not in st.session_state.server_history:
                st.session_state.server_history.append(endpoint_to_connect)

            # --- Create Client and Bridge ---
            st.session_state.client = MCPClient(st.session_state.mcp_endpoint)
            
            temp_bridge = None
            if provider_to_connect == "openai":
                if not openai_key_to_use: raise ValueError("OpenAI API key missing.")
                temp_bridge = OpenAIBridge(st.session_state.client, openai_key_to_use, model=st.session_state.openai_model)
            elif provider_to_connect == "anthropic":
                if not anthropic_key_to_use: raise ValueError("Anthropic API key missing.")
                temp_bridge = AnthropicBridge(st.session_state.client, anthropic_key_to_use, model=st.session_state.anthropic_model)
            elif provider_to_connect == "ollama":
                host = ollama_host_to_use if ollama_host_to_use else None
                
                # Try to fetch Ollama models if not already fetched
                if not st.session_state.ollama_models:
                    try:
                        with st.spinner("Fetching Ollama models..."):
                            models = asyncio.run(fetch_ollama_models(host))
                            if models:
                                st.session_state.ollama_models = models
                                st.success(f"Found {len(models)} Ollama models")
                                
                                # If the selected model is not in the list, use the first available model
                                if ollama_model_to_use not in models and models:
                                    ollama_model_to_use = models[0]
                                    st.session_state.ollama_model = ollama_model_to_use
                                    st.info(f"Selected model '{ollama_model_to_use}' from available models")
                    except Exception as e:
                        st.warning(f"Could not fetch Ollama models: {e}. Using model name as provided.")
                
                temp_bridge = OllamaBridge(st.session_state.client, model=st.session_state.ollama_model, host=host)
            else:
                 raise ValueError(f"Unsupported LLM provider selected: {provider_to_connect}")

            # --- Assign Bridge, Fetch Tools, and Finalize Connection ---
            if temp_bridge:
                st.session_state.llm_bridge = temp_bridge 
                st.session_state.tools = asyncio.run(fetch_tools_async(st.session_state.client)) 
                st.session_state.connected = True
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"Connected to MCP server: {st.session_state.mcp_endpoint}. You can now ask questions."
                })
            else:
                 # Should not be reached if provider check is robust
                 st.error(f"Failed to initialize LLM bridge for provider: {provider_to_connect}") 
                 st.session_state.connected = False # Ensure disconnected state
                 st.session_state.client = None
                 st.session_state.llm_bridge = None
                 st.session_state.tools = []
            
            st.rerun()

        except Exception as e:
            # Handle connection errors
            st.error(f"Error connecting to MCP server: {str(e)}")
            st.session_state.connected = False
            st.session_state.client = None
            st.session_state.llm_bridge = None
            st.session_state.tools = []
            # No rerun here, let error show

# Handle disconnect button
if disconnect_button:
    save_current_conversation() # Save before disconnecting
    # Update connection status
    st.session_state.connected = False
    st.session_state.client = None
    st.session_state.llm_bridge = None
    st.session_state.tools = []

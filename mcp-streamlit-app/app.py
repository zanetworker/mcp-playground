import streamlit as st
import sys
import os
import json
import asyncio
import datetime
import markdown  # Import markdown library
from typing import List, Dict, Any, Optional, Union

# Add parent directory to path to import mcp_sse_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the MCP SSE client
from mcp_sse_client.client import MCPClient
from mcp_sse_client.llm_bridge.openai_bridge import OpenAIBridge
from mcp_sse_client.llm_bridge.anthropic_bridge import AnthropicBridge

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
        margin-bottom: 5px; /* Add some space between buttons */
    }
    .stButton button:hover {
        background-color: #45a049;
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
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []  # List of past conversations
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = 0
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", "")
    }
if "mcp_endpoint" not in st.session_state:
    st.session_state.mcp_endpoint = "http://localhost:8000/sse"
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "openai"
if "server_history" not in st.session_state:
    st.session_state.server_history = []  # List of previously used servers
if "last_interaction_time" not in st.session_state:
    st.session_state.last_interaction_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
        mcp_endpoint = st.text_input(
            "MCP Endpoint URL",
            value=st.session_state.mcp_endpoint,
            key="mcp_endpoint_input",
            help="Enter the full URL for the MCP SSE server. Previously used servers will be remembered."
        )

        st.subheader("LLM Configuration")
        # LLM Provider
        llm_provider = st.selectbox(
            "LLM Provider",
            ["openai", "anthropic"],
            index=0 if st.session_state.llm_provider == "openai" else 1,
            key="llm_provider_select"
        )
        
        # API Keys
        if llm_provider == "openai":
            openai_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.api_keys["openai"],
                key="openai_api_key_input"
            )
        else:
            anthropic_api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.api_keys["anthropic"],
                key="anthropic_api_key_input"
            )
        
        # Save API Keys (Consider security implications)
        # save_api_keys = st.checkbox("Save API Keys (not recommended)", value=False, key="save_api_keys_checkbox")

        st.subheader("Connection Control")
        col1, col2 = st.columns(2)
        with col1:
            # Connect Button - Shortened Text
            connect_button = st.button(
                "Connect",
                key="connect_button",
                help="Connect to the specified server. This will start a new chat."
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
             st.caption(f"LLM: {st.session_state.llm_provider}")


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

# Main content area
if not st.session_state.connected:
    st.info("Please connect to an MCP server using the sidebar.")
else:
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
    
    # Chat input
    if query := st.chat_input("Type your query here..."):
        # Update last interaction time
        st.session_state.last_interaction_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"): st.write(query)
        
        # Process query
        with st.spinner("Processing query..."):
            try:
                llm_result = asyncio.run(st.session_state.llm_bridge.process_query(query))
                
                llm_response = llm_result.get("llm_response")
                tool_call = llm_result.get("tool_call")
                tool_result = llm_result.get("tool_result")
                
                reasoning, final_response = None, None
                if llm_response:
                    if hasattr(llm_response, 'choices') and llm_response.choices: # OpenAI
                        message = llm_response.choices[0].message
                        if message.content:
                            if tool_call: reasoning = message.content
                            else: final_response = message.content
                    elif hasattr(llm_response, 'content'): # Anthropic
                        content_text = "\n".join([c.text for c in llm_response.content if hasattr(c, 'type') and c.type == "text"])
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
            # Save current conversation before switching servers
            save_current_conversation()
            
            # Update session state with potentially new values
            st.session_state.mcp_endpoint = mcp_endpoint
            st.session_state.llm_provider = llm_provider
            # Update the correct API key based on the provider
            if st.session_state.llm_provider == "openai":
                st.session_state.api_keys["openai"] = openai_api_key
            else:
                st.session_state.api_keys["anthropic"] = anthropic_api_key

            # Add new endpoint to history if it's not already there and is a valid URL format (basic check)
            if mcp_endpoint and mcp_endpoint.startswith(("http://", "https://")) and mcp_endpoint not in st.session_state.server_history:
                st.session_state.server_history.append(mcp_endpoint)
            
            # Reset conversation for the new server
            reset_conversation() 
            
            # Create MCP SSE client
            st.session_state.client = MCPClient(st.session_state.mcp_endpoint)
            
            # Create LLM bridge using the correct key
            api_key_to_use = st.session_state.api_keys[st.session_state.llm_provider]
            if not api_key_to_use:
                 raise ValueError(f"API key for {st.session_state.llm_provider} is missing.")

            if st.session_state.llm_provider == "openai":
                st.session_state.llm_bridge = OpenAIBridge(st.session_state.client, api_key_to_use)
            else: # anthropic
                st.session_state.llm_bridge = AnthropicBridge(st.session_state.client, api_key_to_use)
            
            # Define an async helper function to fetch tools
            async def fetch_tools_async(client):
                return await client.list_tools()

            # Get available tools using the helper
            st.session_state.tools = asyncio.run(fetch_tools_async(st.session_state.client))
            
            # Update connection status
            st.session_state.connected = True
            
            # Add system message
            st.session_state.messages.append({
                "role": "system",
                "content": f"Connected to MCP server: {st.session_state.mcp_endpoint}. You can now ask questions."
            })
            
            # Rerun to update UI
            st.rerun()
        
        except Exception as e:
            st.error(f"Error connecting to MCP server: {str(e)}")
            st.session_state.connected = False
            st.session_state.client = None
            st.session_state.llm_bridge = None
            st.session_state.tools = []
            # Don't rerun here, let the error message be displayed

# Handle disconnect button
if disconnect_button:
    save_current_conversation() # Save before disconnecting
    # Update connection status
    st.session_state.connected = False
    st.session_state.client = None
    st.session_state.llm_bridge = None
    st.session_state.tools = []
    
    # Add system message
    st.session_state.messages.append({
        "role": "system",
        "content": "Disconnected from MCP server."
    })
    
    # Rerun to update UI
    st.rerun()

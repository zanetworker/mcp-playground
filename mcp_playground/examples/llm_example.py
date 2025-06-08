"""
Example showing how to use the MCP SSE Client with an LLM.

This script demonstrates how to integrate the MCP client with OpenAI or Anthropic
to enable LLM-driven tool selection and invocation, with a focus on understanding
how the LLM makes its tool selection decisions.
"""

import asyncio
import asyncio
import sys
import os
import json
import textwrap
import asyncio
import sys
import os
import json
import textwrap
import asyncio
import sys
import os
import json
import textwrap
import argparse  # Import argparse
from mcp_playground import MCPClient
# Import all bridges
from mcp_playground.llm_bridge import OpenAIBridge, AnthropicBridge, OllamaBridge
from mcp_playground.format_converters import to_openai_format, to_anthropic_format
# Import model definitions
from mcp_playground.llm_bridge.models import (
    OPENAI_MODELS, DEFAULT_OPENAI_MODEL, 
    ANTHROPIC_MODELS, DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OLLAMA_MODEL # Import Ollama default
)


def print_section(title, content, indent=0):
    """Print a formatted section with a title and content."""
    indent_str = " " * indent
    print(f"\n{indent_str}=== {title} ===")
    if isinstance(content, str):
        for line in content.split('\n'):
            wrapped_lines = textwrap.wrap(line, width=100 - indent)
            for wrapped in wrapped_lines:
                print(f"{indent_str}{wrapped}")
    else:
        print(f"{indent_str}{content}")


def print_tool_summary(tools, formatted_tools=None):
    """Print a summary of the available tools."""
    print_section("Available Tools Summary", f"{len(tools)} tools available:")
    
    for i, tool in enumerate(tools):
        print(f"  {i+1}. {tool.name}: {tool.description}")
        
        # Print key parameters
        required_params = [p for p in tool.parameters if p.required]
        if required_params:
            print("     Required parameters:")
            for param in required_params:
                print(f"       - {param.name} ({param.parameter_type}): {param.description}")
        
        optional_params = [p for p in tool.parameters if not p.required]
        if optional_params:
            print("     Optional parameters:")
            for param in optional_params:
                print(f"       - {param.name} ({param.parameter_type}): {param.description}")
    
    if formatted_tools:
        print("\n  Note: These tools have been formatted for the LLM with proper JSON Schema types.")


def extract_reasoning(llm_response, provider):
    """Extract reasoning from LLM response based on provider."""
    if provider == "openai":
        if hasattr(llm_response.choices[0].message, 'content') and llm_response.choices[0].message.content:
            return llm_response.choices[0].message.content
        return "[No explicit reasoning provided]"
    else:  # anthropic
        text_parts = []
        for content in llm_response.content:
            if content.type == "text":
                text_parts.append(content.text)
        return "\n".join(text_parts) if text_parts else "[No explicit reasoning provided]"


async def main():
    """Run the MCP-LLM integration example with focus on understanding tool selection."""
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="MCP-LLM Integration Example with Tool Selection Analysis")
    
    # Use imported model lists and defaults
    parser.add_argument(
        "--provider", 
        choices=["openai", "anthropic", "ollama"], # Add ollama
        help="Select the LLM provider (openai, anthropic, or ollama). If not provided, you will be prompted."
    )
    parser.add_argument(
        "--openai-model", 
        choices=OPENAI_MODELS, 
        default=DEFAULT_OPENAI_MODEL, 
        help=f"Select the OpenAI model to use (default: {DEFAULT_OPENAI_MODEL}). Choices: {', '.join(OPENAI_MODELS)}"
    )
    parser.add_argument(
        "--anthropic-model", 
        choices=ANTHROPIC_MODELS, 
        default=DEFAULT_ANTHROPIC_MODEL, 
        help=f"Select the Anthropic model to use (default: {DEFAULT_ANTHROPIC_MODEL}). Choices: {', '.join(ANTHROPIC_MODELS)}"
    )
    parser.add_argument(
        "--ollama-model", 
        default=DEFAULT_OLLAMA_MODEL, 
        help=f"Specify the Ollama model name (default: {DEFAULT_OLLAMA_MODEL}). Ensure it's available locally."
    )
    parser.add_argument(
        "--ollama-host", 
        help="Specify the Ollama host URL (e.g., 'http://localhost:11434'). Uses library default if not set."
    )
    parser.add_argument(
        "--endpoint", 
        default=os.environ.get("MCP_ENDPOINT", "http://localhost:8000/sse"), 
        help="MCP SSE endpoint URL (default: http://localhost:8000/sse or MCP_ENDPOINT env var)"
    )
    parser.add_argument("--openai-key", help="OpenAI API key (overrides OPENAI_API_KEY env var)")
    parser.add_argument("--anthropic-key", help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")
    # No API key arg for Ollama
    
    args = parser.parse_args()
    
    print("Starting MCP-LLM Integration Example...")
    
    # --- Configuration Setup ---
    endpoint = args.endpoint
    
    # Determine provider
    provider = args.provider
    if not provider:
        provider = input("Select LLM provider (openai/anthropic): ").strip().lower()

    # Initialize MCP client
    client = MCPClient(endpoint)
    print(f"Connecting to MCP server at: {endpoint}")

    # Setup bridge based on provider
    if provider == "openai":
        api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = input("Enter OpenAI API key: ").strip()
        
        model = args.openai_model
        bridge = OpenAIBridge(client, api_key, model=model)
        print(f"Using OpenAI LLM bridge with model: {model}")
    
    elif provider == "anthropic":
        api_key = args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = input("Enter Anthropic API key: ").strip()
        
        model = args.anthropic_model
        bridge = AnthropicBridge(client, api_key, model=model)
        print(f"Using Anthropic LLM bridge with model: {model}")

    elif provider == "ollama":
        # No API key needed for Ollama
        model = args.ollama_model
        host = args.ollama_host # Will be None if not provided, which is handled by the bridge
        bridge = OllamaBridge(client, model=model, host=host)
        print(f"Using Ollama LLM bridge with model: {model} (Host: {host or 'Default'})")
        # Optional: Add connection check
        if not await bridge.check_connection():
             print(f"Warning: Could not verify connection to Ollama. Ensure it's running and model '{model}' is available.", file=sys.stderr)
    
    else:
        print(f"Unsupported provider: {provider}", file=sys.stderr)
        return

    # --- Tool Fetching and Interaction ---
    print("Fetching tools from server...")
    tools = await bridge.fetch_tools()
    
    # Show tool summary
    print_tool_summary(tools)
    
    # Interactive mode
    print("\nEntering interactive mode. Type 'quit' to exit.")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ("quit", "exit"):
            break
        
        print_section("User Query", query)
        print("Processing query...")
        
        # Get the formatted tools that will be sent to the LLM
        if provider == "openai" or provider == "ollama": # Ollama uses OpenAI format
            formatted_tools = to_openai_format(tools)
        elif provider == "anthropic":
            formatted_tools = to_anthropic_format(tools)
        else:
             formatted_tools = [] # Should not happen due to earlier check

        # Process the query
        result = await bridge.process_query(query)
        
        # Extract and show LLM's reasoning
        # Need to handle different response structures
        llm_response = result["initial_llm_response"]
        reasoning = "[Could not extract reasoning]" # Default
        if provider == "openai":
            if hasattr(llm_response.choices[0].message, 'content') and llm_response.choices[0].message.content:
                reasoning = llm_response.choices[0].message.content
        elif provider == "anthropic":
            text_parts = [c.text for c in llm_response.content if hasattr(c, 'type') and c.type == "text"]
            if text_parts: reasoning = "\n".join(text_parts)
        elif provider == "ollama":
             if isinstance(llm_response, dict) and 'message' in llm_response and llm_response['message'].get('content'):
                 # Check if tool calls exist; if so, reasoning might be empty or just whitespace
                 if not llm_response['message'].get('tool_calls'):
                     reasoning = llm_response['message']['content']
                 # Optionally, could try to extract pre-tool-call text if available, but Ollama structure varies
        
        print_section("LLM Reasoning", reasoning)
        
        # Show tool selection decision
        if result["tool_call"]:
            selected_tool = result["tool_call"]["name"]
            params = result["tool_call"]["parameters"]
            
            # Find the matching tool definition
            tool_def = next((t for t in tools if t.name == selected_tool), None)
            
            print_section("Tool Selection Decision", f"Selected: {selected_tool}")
            if tool_def:
                print(f"  Description: {tool_def.description}")
                
                # Show parameter matching
                print("\n  Parameters provided:")
                for param_name, param_value in params.items():
                    param_def = next((p for p in tool_def.parameters if p.name == param_name), None)
                    if param_def:
                        required = "required" if param_def.required else "optional"
                        print(f"    - {param_name} ({param_def.parameter_type}, {required}): {param_value}")
                        print(f"      Description: {param_def.description}")
            
            # Show how the query maps to the tool selection
            print("\n  Query to Tool Mapping:")
            print(f"    Query: \"{query}\"")
            print(f"    Tool: {selected_tool}")
            print(f"    Key parameters: {', '.join(params.keys())}")
        else:
            print_section("Tool Selection Decision", "No tool was selected by the LLM")
        
        # Show tool result if any
        if result["tool_result"]:
            print_section("Tool Execution Result", 
                         f"Success: {result['tool_result'].error_code == 0}\n" +
                         f"Content: {result['tool_result'].content}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

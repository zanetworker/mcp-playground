"""
Example showing how to use the MCP SSE Client with an LLM.

This script demonstrates how to integrate the MCP client with OpenAI or Anthropic
to enable LLM-driven tool selection and invocation, with a focus on understanding
how the LLM makes its tool selection decisions.
"""

import asyncio
import sys
import os
import json
import textwrap
from mcp_sse_client import MCPClient
from mcp_sse_client.llm_bridge import OpenAIBridge, AnthropicBridge
from mcp_sse_client.format_converters import to_openai_format, to_anthropic_format


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
    print("Starting MCP-LLM Integration Example...")
    
    # Get endpoint
    endpoint = os.environ.get("MCP_ENDPOINT", "http://localhost:8000/sse")
    
    # Initialize MCP client
    client = MCPClient(endpoint)
    
    # Choose LLM provider
    provider = input("Select LLM provider (openai/anthropic): ").strip().lower()
    
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = input("Enter OpenAI API key: ").strip()
        
        bridge = OpenAIBridge(client, api_key)
        print("Using OpenAI LLM bridge")
    
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = input("Enter Anthropic API key: ").strip()
        
        bridge = AnthropicBridge(client, api_key)
        print("Using Anthropic LLM bridge")
    
    else:
        print(f"Unsupported provider: {provider}", file=sys.stderr)
        return
    
    # Fetch tools
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
        if provider == "openai":
            formatted_tools = to_openai_format(tools)
        else:  # anthropic
            formatted_tools = to_anthropic_format(tools)
        
        # Process the query
        result = await bridge.process_query(query)
        
        # Extract and show LLM's reasoning
        reasoning = extract_reasoning(result["llm_response"], provider)
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

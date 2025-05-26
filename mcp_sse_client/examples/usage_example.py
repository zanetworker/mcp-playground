"""
MCP SSE Client Usage Example

This script demonstrates how to use the MCPClient to interact with an MCP endpoint,
list available tools, and invoke a tool with parameters.
"""

import asyncio
import sys
from mcp_sse_client import MCPClient

async def main():
    print("Starting MCPClient example...")
    try:
        # Initialize the client
        print("Initializing client...")
        client = MCPClient("http://localhost:8001/sse")
        
        # List available tools
        print("Listing available tools...")
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
            print("  Parameters:")
            for param in tool.parameters:
                print(f"    - {param.name} ({param.parameter_type}): {param.description}")
        
        # # Invoke a tool
        # print("\nInvoking tool 'convert_document'...")
        # result = await client.invoke_tool(
        #     "convert_document", 
        #     {
        #         "source": "https://arxiv.org/pdf/2404.09982",
        #         "enable_ocr": False
        #     }
        # )
        # print(f"\nTool result: {result.content}")
        # print(f"Error code: {result.error_code}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
    print("Script completed.")

#!/usr/bin/env python3
"""
Quick test script to verify MCP connection and tool calling functionality
"""
import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_sse_client.client import MCPClient
from mcp_sse_client.llm_bridge.openrouter_bridge import OpenRouterBridge

async def test_mcp_connection():
    """Test MCP connection and tool calling"""
    print("🔍 Testing MCP Connection...")
    
    # Test MCP client connection
    client = MCPClient("http://localhost:8001/sse")
    
    try:
        print("📡 Connecting to MCP server...")
        await client.connect()
        print("✅ MCP server connected successfully")
        
        # Get available tools
        tools = await client.list_tools()
        print(f"🔧 Found {len(tools)} tools: {[tool.name for tool in tools]}")
        
        if tools:
            # Test tool calling with OpenRouter bridge
            print("\n🤖 Testing LLM Bridge with tools...")
            
            # You'll need to set your OpenRouter API key
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                print("❌ OPENROUTER_API_KEY environment variable not set")
                return
            
            bridge = OpenRouterBridge(
                api_key=api_key,
                model="openai/gpt-3.5-turbo",
                tools=tools
            )
            
            # Test query that should trigger tool usage
            test_query = "List JIRA issues in the RHOAISTRAT project with component 'llama stack'"
            print(f"📝 Testing query: {test_query}")
            
            result = await bridge.process_query(test_query, [])
            
            if isinstance(result, dict):
                print("📊 Result structure:")
                print(f"  - LLM Response: {bool(result.get('llm_response'))}")
                print(f"  - Tool Call: {bool(result.get('tool_call'))}")
                print(f"  - Tool Result: {bool(result.get('tool_result'))}")
                
                if result.get('tool_call'):
                    tool_call = result['tool_call']
                    print(f"  - Tool Used: {tool_call.get('name', 'Unknown')}")
                
                if result.get('tool_result'):
                    tool_result = result['tool_result']
                    print(f"  - Tool Success: {tool_result.error_code == 0}")
            
            print("✅ Tool calling test completed")
        else:
            print("⚠️ No tools available for testing")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.disconnect()
        print("🔌 Disconnected from MCP server")

if __name__ == "__main__":
    asyncio.run(test_mcp_connection())
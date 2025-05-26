"""
Format converters for transforming MCP tool definitions to various LLM formats.
"""
from typing import List, Dict, Any
from .client import ToolDef, ToolParameter

# Type mapping from Python/MCP types to JSON Schema types
TYPE_MAPPING = {
    "int": "integer",
    "bool": "boolean",
    "str": "string",
    "float": "number",
    "list": "array",
    "dict": "object",
    "boolean": "boolean",
    "string": "string",
    "integer": "integer",
    "number": "number",
    "array": "array",
    "object": "object"
}


def _infer_array_item_type(param: ToolParameter) -> str:
    """Infer the item type for an array parameter based on its name and description.
    
    Args:
        param: The ToolParameter object
        
    Returns:
        The inferred JSON Schema type for array items
    """
    # Default to string items
    item_type = "string"
    
    # Check if parameter name contains hints about item type
    param_name_lower = param.name.lower()
    if any(hint in param_name_lower for hint in ["language", "code", "tag", "name", "id"]):
        item_type = "string"
    elif any(hint in param_name_lower for hint in ["number", "count", "amount", "index"]):
        item_type = "integer"
    
    # Also check the description for hints
    if param.description:
        desc_lower = param.description.lower()
        if "string" in desc_lower or "text" in desc_lower or "language" in desc_lower:
            item_type = "string"
        elif "number" in desc_lower or "integer" in desc_lower or "int" in desc_lower:
            item_type = "integer"
    
    return item_type


def to_openai_format(tools: List[ToolDef]) -> List[Dict[str, Any]]:
    """Convert ToolDef objects to OpenAI function format.
    
    Args:
        tools: List of ToolDef objects to convert
        
    Returns:
        List of dictionaries in OpenAI function format
    """
    
    openai_tools = []
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        # Add properties
        for param in tool.parameters:
            # Map the type or use the original if no mapping exists
            schema_type = TYPE_MAPPING.get(param.parameter_type, param.parameter_type)
            
            param_schema = {
                "type": schema_type,  # Use mapped type
                "description": param.description
            }
            
            # For arrays, we need to specify the items type
            if schema_type == "array":
                item_type = _infer_array_item_type(param)
                param_schema["items"] = {"type": item_type}
            
            openai_tool["function"]["parameters"]["properties"][param.name] = param_schema
            
            # Add default value if provided
            if param.default is not None:
                openai_tool["function"]["parameters"]["properties"][param.name]["default"] = param.default
                
            # Add to required list if required
            if param.required:
                openai_tool["function"]["parameters"]["required"].append(param.name)
                
        openai_tools.append(openai_tool)
    return openai_tools


def to_anthropic_format(tools: List[ToolDef]) -> List[Dict[str, Any]]:
    """Convert ToolDef objects to Anthropic tool format.
    
    Args:
        tools: List of ToolDef objects to convert
        
    Returns:
        List of dictionaries in Anthropic tool format
    """
    
    anthropic_tools = []
    for tool in tools:
        anthropic_tool = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        
        # Add properties
        for param in tool.parameters:
            # Map the type or use the original if no mapping exists
            schema_type = TYPE_MAPPING.get(param.parameter_type, param.parameter_type)
            
            param_schema = {
                "type": schema_type,  # Use mapped type
                "description": param.description
            }
            
            # For arrays, we need to specify the items type
            if schema_type == "array":
                item_type = _infer_array_item_type(param)
                param_schema["items"] = {"type": item_type}
            
            anthropic_tool["input_schema"]["properties"][param.name] = param_schema
            
            # Add default value if provided
            if param.default is not None:
                anthropic_tool["input_schema"]["properties"][param.name]["default"] = param.default
                
            # Add to required list if required
            if param.required:
                anthropic_tool["input_schema"]["required"].append(param.name)
                
        anthropic_tools.append(anthropic_tool)
    return anthropic_tools

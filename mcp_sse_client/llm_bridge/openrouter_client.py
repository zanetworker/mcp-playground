"""
OpenRouter client for unified access to multiple LLM providers.
"""
import requests
import asyncio
from typing import List, Dict, Any, Optional


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(self, api_key: str, site_url: Optional[str] = None, site_name: Optional[str] = None):
        """Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
            site_url: Optional site URL for rankings
            site_name: Optional site name for rankings
        """
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.base_url = "https://openrouter.ai/api/v1"
    
    async def fetch_models(self) -> List[Dict[str, Any]]:
        """Fetch all available models from OpenRouter.
        
        Returns:
            List of model dictionaries with metadata
        """
        try:
            # Use requests for now, can be made async later if needed
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("data", [])
            
        except Exception as e:
            print(f"Error fetching OpenRouter models: {e}")
            return []
    
    async def fetch_top_models_by_provider(self, provider: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch top N most popular models for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'google')
            limit: Maximum number of models to return
            
        Returns:
            List of top models for the provider
        """
        all_models = await self.fetch_models()
        
        # Filter by provider prefix (models are already sorted by popularity)
        provider_models = [
            model for model in all_models 
            if model["id"].startswith(f"{provider}/")
        ]
        
        # Return top N models
        return provider_models[:limit]
    
    def get_extra_headers(self) -> Dict[str, str]:
        """Get extra headers for OpenRouter requests.
        
        Returns:
            Dictionary of extra headers
        """
        headers = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        return headers


def format_model_display(model: Dict[str, Any]) -> Dict[str, Any]:
    """Format model information for display in UI.
    
    Args:
        model: Model dictionary from OpenRouter API
        
    Returns:
        Formatted model information
    """
    name = model.get("name", model.get("id", "Unknown"))
    model_id = model["id"]
    pricing = model.get("pricing", {})
    context_length = model.get("context_length", "Unknown")
    description = model.get("description", "")
    
    # Format pricing (convert to per 1M tokens)
    try:
        prompt_price = float(pricing.get("prompt", 0)) * 1000000
        completion_price = float(pricing.get("completion", 0)) * 1000000
        pricing_str = f"${prompt_price:.2f}/${completion_price:.2f} per 1M"
    except (ValueError, TypeError):
        pricing_str = "Pricing unavailable"
    
    # Format context length
    if isinstance(context_length, (int, float)) and context_length > 0:
        context_str = f"{int(context_length):,} ctx"
    else:
        context_str = "Unknown ctx"
    
    # Create display string
    display_name = f"{name} | {pricing_str} | {context_str}"
    
    return {
        "display": display_name,
        "id": model_id,
        "name": name,
        "description": description,
        "pricing": pricing,
        "context_length": context_length
    }
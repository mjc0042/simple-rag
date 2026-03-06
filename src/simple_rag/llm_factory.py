"""LLM Factory for creating LangChain chat models based on model name."""
from typing import Any

from langchain.chat_models import init_chat_model


def create_chat_model(model_name: str, api_key: str, **kwargs: Any):
    """
    Create a LangChain chat model based on the model name.
    
    Args:
        model_name: The name of the model (e.g., gpt-4o-mini, deepseek-exp, claude-sonnet-4-20250514)
        api_key: The API key for the model provider
        **kwargs: Additional arguments to pass to the chat model
    
    Returns:
        A LangChain chat model instance
    """
    # Default temperature
    temperature = kwargs.get("temperature", 0)
    
    return init_chat_model(model=model_name,
                           configurable_fields=("api_key"),
                           temperature=temperature,
                           api_key=api_key
            )

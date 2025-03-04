import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from models.models import ResponseDict, HandbookChunk
from components.system_prompts import get_chatbot_prompt

def get_model(provider: str, name: str):
    if provider.lower() == "open-ai":
        return OpenAIModel(name, api_key=os.environ.get("OPENAI_API_KEY"))
    elif provider.lower() == "google":
        return GeminiModel(name, api_key=os.environ.get("GEMINI_API_KEY"))
    elif provider.lower() == "anthropic":
        return AnthropicModel(name, api_key=os.environ.get("ANTHROPIC_API_KEY"))
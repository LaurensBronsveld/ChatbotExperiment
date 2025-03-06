import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from config import settings


def get_model():
    if settings.LLM_PROVIDER.lower() == "open-ai":
        return OpenAIModel(settings.LLM_MODEL, api_key=settings.OPENAI_API_KEY)
    elif settings.LLM_PROVIDER.lower() == "google":
        return GeminiModel(settings.LLM_MODEL, api_key=os.environ.get("GEMINI_API_KEY"))
    elif settings.LLM_PROVIDER.lower() == "anthropic":
        return AnthropicModel(settings.LLM_MODEL, api_key=os.environ.get("ANTHROPIC_API_KEY"))
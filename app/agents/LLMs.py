import os
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel

# load environment variables
load_dotenv()

def get_model(provider: str, name: str):
    if provider.lower() == "open-ai":
        return OpenAIModel(name, api_key=os.environ.get("OPENAI_API_KEY"))
    elif provider.lower() == "google":
        return GeminiModel(name, api_key=os.environ.get("GEMINI_API_KEY"))
    elif provider.lower() == "anthropic":
        return AnthropicModel(name, api_key=os.environ.get("ANTHROPIC_API_KEY"))
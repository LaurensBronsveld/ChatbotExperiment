import logging
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from app.core.config import settings


def get_model():
    """
    Retrieves an LLM model instance based on the configuration settings.

    Reads `settings.LLM_PROVIDER` to determine whether to initialize an
    OpenAI, Google Gemini, or Anthropic model using `settings.LLM_MODEL`.

    Returns:
        OpenAIModel | GeminiModel | AnthropicModel | None: An instance of the specified
        LLM provider's model. Returns default OpenAI model if provider is not recognised.
    """

    if settings.LLM_PROVIDER.lower() == "open-ai":
        return OpenAIModel(settings.LLM_MODEL)
    elif settings.LLM_PROVIDER.lower() == "google":
        return GeminiModel(settings.LLM_MODEL)
    elif settings.LLM_PROVIDER.lower() == "anthropic":
        return AnthropicModel(settings.LLM_MODEL, api_key=settings.ANTHROPIC_API_KEY)
    else:
        logging.warning("LLM provider not recognized. Returning default OpenAI model.")
        return OpenAIModel("gpt-4o")

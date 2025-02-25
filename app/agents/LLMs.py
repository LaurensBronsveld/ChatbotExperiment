import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from models.chat import ResponseModel, QueryModel

class OpenAIAgent:
    model = OpenAIModel('gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))
    agent = Agent(model, result_type=ResponseModel)    

class GeminiAgent:
    model = GeminiModel('gemini-2.0-flash-exp', api_key=os.environ.get("GEMINI_API_KEY"))
    agent = Agent(model, result_type=ResponseModel)    

class AnthropicAgent:
    model = AnthropicModel('claude-3-5-sonnet-latest', api_key=os.environ.get("ANTHROPHIC_API_KEY"))
    agent = Agent(model, result_type=ResponseModel)    
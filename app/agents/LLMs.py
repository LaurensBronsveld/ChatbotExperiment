import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from models.models import ResponseDict, HandbookChunk
from components.system_prompts import get_chatbot_prompt



    
# def ddg_search(self, ctx: RunContext, query: str):
#         try:
#             with DDGS() as ddgs:
#                 results = [r for r in ddgs.text(query, max_results=5)]
#                 return {"results": results}
#         except Exception as e:
#             return {"error": str(e)}

class OpenAIAgent:


    def search_database(self, ctx: RunContext, query: str):
            try:
                results = self.table.search(query, vector_column_name='vector').limit(5).to_pydantic(HandbookChunk)
                json_results = []
                for chunk in results:
                    
                    chunk_dict = {
                        "chunk_id": chunk.chunk_id,
                        "source_url": chunk.source_url,
                        "chunk": chunk.chunk
                    }
                    json_results.append(chunk_dict)
                return json_results
            except Exception as e:
                return{"error": str(e)}

    def __init__(self):
        self.model = OpenAIModel('gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))
        self.agent = Agent(self.model, result_type=ResponseDict, system_prompt=get_chatbot_prompt())
        self.agent.tool(self.search_database)
       

class GeminiAgent:
    model = GeminiModel('gemini-2.0-flash-exp', api_key=os.environ.get("GEMINI_API_KEY"))
    agent = Agent(model, result_type=ResponseDict)    

class AnthropicAgent:
    model = AnthropicModel('claude-3-5-sonnet-latest', api_key=os.environ.get("ANTHROPHIC_API_KEY"))
    agent = Agent(model, result_type=ResponseDict)    
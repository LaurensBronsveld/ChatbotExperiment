from __future__ import annotations as _annotations  # Should be first import

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List
import json
import os
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

import asyncio
from duckduckgo_search import DDGS
from openai import OpenAI
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from models.chat import ResponseModel, QueryModel
from agents.LLMs import OpenAIAgent

# If you're using logging for debugging
import logging

from DatabaseManager import DatabaseManager



# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

class HandbookChunk(LanceModel):
    chunk_id: str | None = None
    file: str | None = None
    chunk: str | None = None


class Assistant_Agent():

    def search_database(self, ctx: RunContext, query: str):
            try:
                
                # client = OpenAI()
                # query_embedding = client.embeddings.create(
                # input=query,
                # model="text-embedding-3-large"
                # ).data[0].embedding

                results = self.table.search(query, vector_column_name='vector').limit(5).to_pydantic(HandbookChunk)
                json_results = []
                for chunk in results:
                    
                    chunk_dict = {
                        "chunk_id": chunk.chunk_id,
                        "file": chunk.file,
                        "chunk": chunk.chunk
                    }
                    json_results.append(chunk_dict)
                return json_results
            except Exception as e:
                return{"error": str(e)}
    
    def ddg_search(self, ctx: RunContext, query: str):
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=5)]
                return {"results": results}
        except Exception as e:
            return {"error": str(e)}

    def __init__(self, db: DatabaseManager):
        
        self.model = OpenAIModel('gpt-4o',
                                 api_key=os.environ.get("OPENAI_API_KEY"))
        
        self.agent = Agent(self.model)    
        #self.agent = OpenAIAgent.agent
        self.history = [
                        {"role": "system", 
                         "content": """
                        You are a helpful assistant who provides answers to questsions related to the Gitlab Handbook. 
                        You have access to an embedded database containing the whole Gitlab Handbook
                        Please use the "search_database" to retrieve relevant information from the Gitlab Handbook to help answer questions.
                        Your workflow should go like this:
                        1. use 'search_database' to retrieve relevant information from your database to help answer the question
                        2. Answer the question as detailed as possible while referring to the handbook pages.
                        3. List the specific pages and sections used as sources including an url to the page when possible.
                        
                        You can not use your internal knowledge
                        You must use the search_database tool to answer Gitlab Handbook related questions. Please explain what went wrong if you cant retrieve relevant information using the RAG_tool
                        """}
                    ]

        self.tokens = {'request': 0, 'response': 0, 'total': 0}
        self.dbmanager = db
        self._table = None

        self.agent.tool(self.search_database)
        # self.agent.tool(self.ddg_search)
        
    @property
    def table(self):
        if self._table is None:
            try:
                self._table = self.dbmanager.get_table('embedded_handbook')
            except Exception as e:
                print(f"error accesssing table: {str(e)}")
                raise
        return self._table
    
    class QueryModel(BaseModel):
        query: str
        use_rag: bool = True
        use_ddrg: bool = False

    async def generate_response(self, query: str, use_rag: bool, use_ddrg: bool):
        self.history.append({'role': "user", "content": query})
        try:
            response = await self.agent.run(str(self.history))
            return response.data
        except Exception as e:
            return{"error": str(e)}
        
    async def generate_response_stream(self, query: str, use_rag: bool, use_ddrg: bool):
        
        self.history.append({'role': "user", "content": query})

        complete_response = ""  # Store the response text as it is streamed
        sources = ["test"]  # Store the sources as they are identified

        try:
            async with self.agent.run_stream(str(self.history)) as result:
                async for message in result.stream():

                    new_content = message[len(complete_response):]
                    complete_response = message  
                    if new_content:
                        yield new_content
                    
                # yield json.dumps({'content': '', 'sources': sources})
                self.history.append({'role': "assistant", "content": complete_response})
        except Exception as e:
            yield json.dumps({"error": str(e)})

db_manager = DatabaseManager("./data/lancedb")
assistant = Assistant_Agent(db_manager)

async def main():
    
    response = await assistant.generate_response("What is GitLab's approach to paid time off (PTO)", True, False)
    print(response)

      # Wait for all pending tasks to complete (IMPORTANT!)
    pending = asyncio.all_tasks()
    while pending:  # Check if there are any pending tasks.
        await asyncio.gather(*pending) # * unpacks the set of tasks.

if __name__ == "__main__":
    asyncio.run(main())
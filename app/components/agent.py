from __future__ import annotations as _annotations  # Should be first import

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List
import json
import os
import lancedb
import re

import uuid
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

import asyncio
from duckduckgo_search import DDGS
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from components.system_prompts import get_chatbot_prompt

from models.models import *
from agents.LLMs import get_model


import logging

from components.DatabaseManager import DatabaseManager



# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

# Silence specific noisy libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logging.getLogger("pydantic_ai").setLevel(logging.WARNING)




class Assistant_Agent():


    def search_database(self, ctx: RunContext, query: str):
            try:
                results = self.table.search(query, vector_column_name='vector').limit(5).to_pydantic(HandbookChunk)
                json_results = []
                id = 1

                for chunk in results:
                    chunk_dict = {
                        "id": id,
                        "source_url": chunk.source_url,
                        "chunk": chunk.chunk
                    }
                    json_results.append(chunk_dict)
                    id += 1
                return json_results
            except Exception as e:
                return{"error": str(e)}
    

    def __init__(self, db: DatabaseManager, model_provider, model_name: str):
        
        self.model = get_model(model_provider, model_name)
        
        self.agent = Agent(self.model, result_type=ResponseDict, system_prompt= get_chatbot_prompt())    
   
        self.history = []
        self.dbmanager = db
        self._table = None

        self.agent.tool(self.search_database)
        # self.agent.tool(self.ddg_search)
        
    @property
    def table(self):
        if self._table is None:
            try:
                self._table = self.dbmanager.get_table("embedded_handbook_with_urls")
            except Exception as e:
                print(f"error accesssing table: {str(e)}")
                raise
        return self._table
    
  
    def get_tool_results(self, result, tool_name):
        content = []
        sources = []
        
        # get resuls from tool call out of Result object
        for message in result.all_messages():
                    
                    for part in message.parts:
                        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name:
                            content.extend(part.content)
                            #sources.append(f"{part.content['id']}. {part.content['source_url']}")

        # create source objects 
        for source in content: 
            url_regex = r"^(https?:\/\/|www\.)\S+$"   # regex which matches most urls starting with http(s)// or www.
            uri_regex = r"^(?:[a-zA-Z]:\\|\/)[^\s]*$" # regex which matches absolute file paths in windows and unix systems
            # check type of source (rough implementation, probably better to do this while building database)
           
            if re.match(url_regex, source['source_url']):
                sources.append(SourceDict(id = source['id'], type = 'url', url=source["source_url"], used=False))
            elif re.match(uri_regex, source['source_url']):
                sources.append(SourceDict(id = source["id"], type = 'file', uri=source["source_url"], used=False))
            else:
                sources.append(SourceDict(id = source['id'], type = 'snippet', text="some text", used=False))
        
        return sources

   

    async def generate_response_stream(self, request: RequestModel):
        self.history.append({'role': "user", "content": request.user['question']})

        complete_content = ""  # Store the response text as it is streamed
        new_content = ""
        
        # if session_id is null, generate one
        if not request.metadata['session_id']:
            request.metadata['session_id'] = str(uuid.uuid4())
    
        try:
            async with self.agent.run_stream(str(self.history)) as result:
                
                sources = self.get_tool_results(result, 'search_database')
                
                async for structured_result, last in result.stream_structured(debounce_by=0.01):            
                    try:
                        chunk = await result.validate_structured_result(
                            structured_result, allow_partial=not last
                        )
                        
                        #determine new content
                        if chunk.get('content'):
                            content = chunk.get('content')
                            new_content = content[len(complete_content):]
                            complete_content = content
                        # determine used sources
                        cite_regex = r"\[@(\d+)\]"   #regex which matches citations in this format [@X], with X being any number
                        citations = re.findall(cite_regex, complete_content)
                    
                        for source in sources:
                            if str(source.get('id')) in citations:
                                source['used'] = True

                        response = ResponseDict(
                            content=new_content, 
                            sources=sources,
                            tools_used = chunk.get("tools_used"),
                            able_to_answer=chunk.get("able_to_answer"),
                            question_classification= chunk.get('question_classification'),
                            session_id=request.metadata['session_id'],
                            trace_id=None,
                            share_token="",
                            follow_up_questions=chunk.get('follow_up_questions'))
                        
                        yield(json.dumps(response).encode('utf-8') + b'\n')

                    except ValidationError as exc:
                        if all(
                            e['type'] == 'missing' and e['loc'] == ('response',)
                            for e in exc.errors()
                        ):
                            continue
                        else:
                            raise
                   
                
                
                logging.debug(json.dumps(response).encode('utf-8') + b'\n')
                logging.debug(complete_content)
                self.history.append({'role': "assistant", "content": complete_content})
                

        except Exception as e:
            yield json.dumps({"error": str(e)})

# db_manager = DatabaseManager("./data/lancedb")
# assistant = Assistant_Agent(db_manager)

# async def main():
    
#     response = await assistant.generate_response("What is GitLab's approach to paid time off (PTO)", True, False)
#     print(response)

#       # Wait for all pending tasks to complete (IMPORTANT!)
#     pending = asyncio.all_tasks()
#     while pending:  # Check if there are any pending tasks.
#         await asyncio.gather(*pending) # * unpacks the set of tasks.

# if __name__ == "__main__":
#     asyncio.run(main())
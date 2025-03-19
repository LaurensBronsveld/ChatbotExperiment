from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List
import json
import os
import httpx
import re
import uuid
import requests
from pydantic import ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ToolReturnPart, ToolCallPart, ModelMessage, ModelMessagesTypeAdapter
from agents.system_prompts import get_chatbot_prompt
from models.models import *
from models.SQL_models import *
from agents.LLMs import get_model
from components.DatabaseManager import get_session
from api.history.history import get_history
from api.tools.tools import search_database
from config import settings
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import cohere
from lancedb.rerankers import CohereReranker
import logging
from sqlalchemy.orm import declarative_base, sessionmaker, Session, scoped_session
import openai
import numpy as np


# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

# Silence specific noisy libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logging.getLogger("pydantic_ai").setLevel(logging.WARNING)



langfuse = Langfuse(
            public_key = settings.LANGFUSE_PUBLIC_KEY,
            secret_key = settings.LANGFUSE_SECRET_KEY,
            host = settings.LANGFUSE_HOST
        )

class BaseAgent():

    def __init__(self, language: str):
        
        self.language = language
        self.tools = []

    def get_chat_history(self, session_id: str):
        try:
            history = []
            results = get_history(session_id = session_id, show_system_calls=True)

            if results:

                for message in results.messages:
                    history.append({'role': message.role.value, "content": message.content})
                return history
            else:
                return None
        except Exception as e:
            logging.error(f'error retrieving chat history: {e}')
    
    def update_chat_history(self, db: object, session_id: UUID, message: MessageModel):
        try:
            #check if session with given session_id exists

            # add new message to database
            # for now it is assumed that the message matches the langauge of the assistant 
            # we might detect this manually in the future
            new_chat_message = ChatMessage(
                session_id = session_id,
                role = message.role,
                message = message.model_dump_json(),
                language = self.language        
            )
            db.add(new_chat_message)
            db.commit()
        except Exception as e:
            logging.error(f"something went wrong updating history : {e}")
       
       
    def use_search_tool(self, ctx: RunContext, query: str, tool_call_atempt: int = 0, limit: int = 5):
        # create url for tool
        url = f"{settings.API_URL}/api/search/"
        result = requests.post(
            f"{url}",
            json={  
                "query": query, 
                "tool_call_attempt": tool_call_atempt,
                "limit": limit
                }
            )                  
        data = result.json()
        return data

    def get_tool_results(self, ctx: RunContext, result: object, tool_name: str, db, session_id):
        content = []
        sources = []
        tools = []
        logging.debug(result)
        # get resuls from tool call out of Result object
        for message in result.all_messages():
                    for part in message.parts:
                        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name:

                            content.extend(part.content)
                            tools.append(part.tool_name)
                            
                            system_message = MessageModel(
                                role = ChatRole.SYSTEM,
                                content = f"Called tool: {tool_name}. Results: {part.content}")
                                
                            self.update_chat_history(db, session_id, system_message)


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
        
        return sources, tools

    @observe(capture_input=True, capture_output=True, as_type="generation", name="chatbot response")
    async def process_answer(self, prompt: list, session_id: str, db):
        response = None
        old_content = ""
            
        trace_id = langfuse_context.get_current_trace_id()
        logging.debug(langfuse_context.get_current_trace_url())
            
        model = get_model()        
        agent = Agent(model, result_type=ResponseModel, system_prompt= get_chatbot_prompt(self.language))    
        agent.tool(self.use_search_tool)
        history = self.get_chat_history(session_id)
      
        async with agent.run_stream(str(history)) as result:

            sources, tools_used = self.get_tool_results(self, result = result, tool_name= 'use_search_tool', db = db, session_id= session_id)
                
            metadata = ResponseMetadata(
                sources = sources,
                tools_used = tools_used,
                session_id = str(session_id),
                trace_id = trace_id) 
            yield(metadata.model_dump_json())      
           
                       
            async for structured_result, last in result.stream_structured(debounce_by=0.01): 
                        
                try:
                    chunk = await result.validate_structured_result(
                    structured_result, allow_partial=not last
                    )                        
                        
                    content = chunk.content
                    if not last:
                        if content != old_content:
                            old_content = content
                            # create response object
                            response = ResponseModel(
                                content=chunk.content, 
                                )    
                            yield(response.model_dump_json())   
                    else: 
                        # create response object
                        response = ResponseModel(
                            content=content, 
                            able_to_answer=chunk.able_to_answer,
                            question_classification= chunk.question_classification,
                            follow_up_questions=chunk.follow_up_questions)
                        yield(response.model_dump_json())
                except ValidationError as exc:
                    if all(e["type"] == "missing" for e in exc.errors()):
                        logging.warning(f"Missing field during streaming: {exc}") #Log the warning.
                        continue
                    else:
                        raise
                except httpx.ReadError as e:
                    logging.error(f"Streaming interrupted: {e}")
                    break  # Stop streaming if connection is lost
                
                    
            assistant_message = MessageModel(
                    role = ChatRole.ASSISTANT,
                    content = content)
                                
            self.update_chat_history(db, session_id, assistant_message)

    async def generate_response_stream(self, request: RequestModel, session_id: UUID = None):

        # get database session
        db_generator = get_session()
        db = next(db_generator)
        
        # start langfuse trace
        trace = langfuse.trace(
            name = "chat_request"
        )

        # add user question to history
        user_message = MessageModel(
            role = ChatRole.USER,
            content = request.user['question'],
            context = request.user['context'])
        self.update_chat_history(db, session_id, user_message)
        
        # get streaming response from agent
        retries = 1
        old_chunk = ""
        for attempt in range(retries):
            try:       
                async for chunk in self.process_answer(request.user['question'], session_id, db):
                    if chunk is not old_chunk:
                        old_chunk = chunk
                        yield chunk
                break
            except Exception as e:
                if attempt < retries-1:
                    logging.error(f"Error: {e} occured while streaming response, repeating attempt")
                    continue
                else:
                    logging.error(f'something went wrong generating the response: error: {e}')
                    db.rollback() # rollback changes to the database
       
        # commit and close database session
        db.close()
        
    def generate_response(self, request: RequestModel):
        
        # if session_id is null, generate one
        if not request.metadata['session_id']:
            request.metadata['session_id'] = str(uuid.uuid4())
            
        else:
            # if session_id exists, retrieve chat history
            history = self.get_chat_history(request.metadata['session_id'])
            session_id = request.metadata['session_id']
            history = json.loads(history)
        
        # add user question to history
        history.append({'role': "User", "content": request.user['question']})

        response = self.agent.run_sync(history)
        history.append({'role': "Assistant", "content": response.content})
        return response
 

# db_manager = DatabaseManager()
# assistant = Assistant(db_manager, 'nl')

# def main():
#     request = {"metadata": {"language": "nl",
#                         "session_id": "",
#                         "tools": [{ "name": "HR", "enabled": True }]
#                     }, 
#                     "user": {
#                         "question": "What is GitLab's approach to paid time off (PTO)",
#                         "context": [
#                             { "type": "file", "URL": "" }, 
#                             { "type": "snippet", "text": ""},
#                             { "type": "url", "url": "https://example.com" }
#                         ]
#                     }
#                     }
#     response = assistant.generate_response(request)
#     print(response)


# if __name__ == "__main__":
#     main()
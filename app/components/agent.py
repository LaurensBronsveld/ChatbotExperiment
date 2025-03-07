from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List
import json
import os
import httpx
import re
import uuid
import asyncio
from pydantic import ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ToolReturnPart, ToolCallPart
from components.system_prompts import get_chatbot_prompt
from models.models import *
from agents.LLMs import get_model
from components.DatabaseManager import DatabaseManager
from config import settings
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv
import logging

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



class Assistant_Agent():


    def search_database(self, ctx: RunContext, query: str):
            try:
                results = self.hand_book_table.search(query, vector_column_name='vector').limit(5).to_pydantic(HandbookChunk)
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
    

    def __init__(self, db: DatabaseManager, language: str):
        
        self.model = get_model()
        
        self.agent = Agent(self.model, result_type=ResponseDict, system_prompt= get_chatbot_prompt(language))    
   
        self.history = []
        self.dbmanager = db
        self._handbook_table = None
        self._history_table = None

        self.agent.tool(self.search_database)

        
        
        
    @property
    def hand_book_table(self):
        if self._handbook_table is None:
            try:
                self._handbook_table = self.dbmanager.get_table(settings.HANDBOOK_TABLE)
            except Exception as e:
                print(f"error accesssing table: {str(e)}")
                raise
        return self._handbook_table
    
    @property
    def history_table(self):
        if self._history_table is None:
            try:
                self._history_table = self.dbmanager.get_table(settings.HISTORY_TABLE)
            except Exception as e:
                print(f"error accesssing table: {str(e)}")
                raise
        return self._history_table
    
    def get_chat_history(self, session_id: str):
        try:
            results = self.history_table.search().where(f"session_id = '{session_id}'").limit(1).to_pydantic(ChatHistory)
            if results:
                return results[0].history, results[0].share_token
            else:
                return None
        except Exception as e:
            logging.error(f'error retrieving chat history: {e}')
    
    def update_chat_history(self, session_id: str, share_token: str, new_history: str):
        try:
            self.dbmanager.update_chat_history(table_name="history_table", session_id=session_id, share_token=share_token, new_history=new_history)
        except Exception as e:
            logging.error(f"something went wrong updating history : {e}")
       
    def get_tool_results(self, result, tool_name):
        content = []
        sources = []
        tools = []
        
        # get resuls from tool call out of Result object
        for message in result.all_messages():
                    for part in message.parts:
                        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name:
                            content.extend(part.content)
                        elif isinstance(part, ToolCallPart):
                            tools.append(part.tool_name)
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
        
        return sources, tools

    @observe(capture_input=True, capture_output=True, as_type="generation", name="chatbot response")
    async def process_answer(self, history: str, session_id: str, share_token: str, retries: int = 3):
            response = None
            old_content = ""
            
            trace_id = langfuse_context.get_current_trace_id()
            logging.debug(langfuse_context.get_current_trace_url())
            
          
            async with self.agent.run_stream(str(history)) as result:
                        
                sources, tools_used = self.get_tool_results(result, 'search_database')
                metadata = ResponseMetadata(
                    sources = sources,
                    tools_used = tools_used,
                    session_id = session_id,
                    trace_id = trace_id,
                    share_token = share_token) 
                yield(json.dumps(metadata).encode('utf-8') + b'\n')      
                logging.debug(json.dumps(metadata).encode('utf-8') + b'\n')
                       
                async for structured_result, last in result.stream_structured(debounce_by=0.01): 
                        
                    try:
                        chunk = await result.validate_structured_result(
                        structured_result, allow_partial=not last
                        )                        
                                    
                        # #determine new content. This should happen in front end
                        # if chunk.get('content'):
                        #     content = chunk.get('content')
                        #     new_content = content[len(complete_content):]
                        #     complete_content = content
                        content = chunk.get('content')
                        if not last:
                            if content != old_content:
                                old_content = content
                                # create response object
                                response = ResponseDict(
                                    content=chunk.get('content'), 
                                    )    
                                yield(json.dumps(response).encode('utf-8') + b'\n')   
                        else: 
                            # # determine used sources
                            # cite_regex = r"\[@(\d+)\]"   #regex which matches citations in this format [@X], with X being any number
                            # citations = re.findall(cite_regex, content)
                                    
                            # for source in sources:
                            #     if str(source.get('id')) in citations:
                            #         source['used'] = True
                            
                            # create response object
                            response = ResponseDict(
                                content=content, 
                                sources=sources,
                                tools_used = chunk.get("tools_used"),
                                able_to_answer=chunk.get("able_to_answer"),
                                question_classification= chunk.get('question_classification'),
                                session_id=session_id,
                                trace_id= trace_id,
                                share_token=share_token,
                                follow_up_questions=chunk.get('follow_up_questions'))
                            yield(json.dumps(response).encode('utf-8') + b'\n')
                        
                        # send response to front end
                        

                    except httpx.ReadError as e:
                        logging.error(f"Streaming interrupted: {e}")
                        break  # Stop streaming if connection is lost
                    except ValidationError as exc:
                        if all(
                            e['type'] == 'missing' and e['loc'] == ('response',)
                                        for e in exc.errors()
                            ):
                            continue
                        else:
                            raise
                    

            history.append({'role': "assistant", "content": content})
            self.update_chat_history(session_id, share_token, json.dumps(history))

                
    async def generate_response_stream(self, request: RequestModel):
        # define variables
        history = []
        session_id = ""
        share_token = ""


        # start langfuse trace
        trace = langfuse.trace(
            name = "chat_request"
        )

        # if session_id is null, generate one
        if not request.metadata['session_id']:
            logging.debug("test")
            request.metadata['session_id'] = str(uuid.uuid4())
            share_token = str(uuid.uuid4())
        else:
            # if session_id exists, retrieve chat history
            history, share_token = self.get_chat_history(request.metadata['session_id'])
            session_id = request.metadata['session_id']
            history = json.loads(history)
            
        # add user question to history
        history.append({'role': "user", "content": request.user['question']})
        
        # get streaming response from agent
        retries = 3
        old_chunk = ""
        for attempt in range(retries):
            try:       
                async for chunk in self.process_answer(history, session_id, share_token):
                    if chunk is not old_chunk:
                        old_chunk = chunk
                        yield chunk
                break
            except Exception as e:
                if attempt < retries-1:
                    logging.error(f"Error: {e} occured while streaming response, repeating attempt")
                    continue
                else:
                    logging.error('something went wrong generating the response')
        

            
        # async def process_answer(retries: int = 3):
        #     for attempt in range(retries):
        #         try:
        #             async with self.agent.run_stream(str(history)) as result:
                        
        #                 sources = self.get_tool_results(result, 'search_database')
                        
        #                 async for structured_result, last in result.stream_structured(debounce_by=0.01):            
        #                     try:
        #                         chunk = await result.validate_structured_result(
        #                             structured_result, allow_partial=not last
        #                         )
                                
        #                         #determine new content
        #                         if chunk.get('content'):
        #                             content = chunk.get('content')
        #                             new_content = content[len(complete_content):]
        #                             complete_content = content

        #                         # determine used sources
        #                         cite_regex = r"\[@(\d+)\]"   #regex which matches citations in this format [@X], with X being any number
        #                         citations = re.findall(cite_regex, complete_content)
                            
        #                         for source in sources:
        #                             if str(source.get('id')) in citations:
        #                                 source['used'] = True

        #                         # create response object
        #                         response = ResponseDict(
        #                             content=new_content, 
        #                             sources=sources,
        #                             tools_used = chunk.get("tools_used"),
        #                             able_to_answer=chunk.get("able_to_answer"),
        #                             question_classification= chunk.get('question_classification'),
        #                             session_id=request.metadata['session_id'],
        #                             trace_id= None,
        #                             share_token=share_token,
        #                             follow_up_questions=chunk.get('follow_up_questions'))
                                
        #                         yield(json.dumps(response).encode('utf-8') + b'\n')

        #                     except httpx.ReadError as e:
        #                         logging.error(f"Streaming interrupted: {e}")
        #                         break  # Stop streaming if connection is lost
        #                     except ValidationError as exc:
        #                         if all(
        #                             e['type'] == 'missing' and e['loc'] == ('response',)
        #                             for e in exc.errors()
        #                         ):
        #                             continue
        #                         else:
        #                             raise
                    

        #             history.append({'role': "assistant", "content": complete_content})
        #             self.update_chat_history(request.metadata['session_id'], share_token, json.dumps(history))
        #             return
        #         except Exception as e:
        #             if attempt < retries - 1:
        #                 await asyncio.sleep(2)
        #             else:       
        #                 logging.error(f"something went wrong with streaming the response: {e}")
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
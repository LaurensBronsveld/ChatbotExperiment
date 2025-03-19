from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List
import json
import os
import httpx
import re
import uuid
from pydantic import ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ToolReturnPart, ToolCallPart
from agents.system_prompts import get_chatbot_prompt
from models.models import *
from models.SQL_models import *
from agents.LLMs import get_model
from components.DatabaseManager import get_session
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

class Assistant():

    def __init__(self, language: str):
        
        self.language = language
        self._handbook_table = None
        self._history_table = None

        self.agent.tool(self.search_database)
    
    def search_database(self, ctx: RunContext, query: str, tool_call_attempt: int, limit: int = 3):
        """
        Searches the handbook database using the provided query for relevant chunks of information.
        
        This method performs a vector search on the LanceDB database containing embedded chunks of the Gitlab Handbook.
        It returns the top X matching results from the database in the form of a JSON list of dictionaries with ID, source and content
        Args:
            ctx (RunContext): The context of the current run, providing access to dependencies and state.
            query (str): The search query to use against the handbook database.
            tool_call_attempt (int): The attempt number of the tool call, used to generate unique IDs for results. First attempt is 0.

        Returns:
            list[dict] or dict: A list of dictionaries, where each dictionary represents a search result
                            containing the 'id', 'source_url', and 'chunk'. 
        """ 
        #setup
        json_results = []
        docs = []
        id = tool_call_attempt * limit # set starting id based on how many times tool has been called before
    
        try:
            #initialise Cohere reranker
            co = cohere.Client(settings.COHERE_API_KEY)
           
            db_generator = get_session()
            db = next(db_generator)
            # perform vector search on database
            query_embedding = openai.embeddings.create(
                input = query,
                model=settings.EMBED_MODEL
            ).data[0].embedding

            query_vector = np.array(query_embedding).tolist()
            
            results = (
                db.query(Chunk)
                .order_by(Chunk.embedding.l2_distance(query_vector))  # L2 distance for similarity
                .limit(100)
                .all()
            )

            #get list of strings to rerank with Cohere
            for chunk in results:
                docs.append(chunk.chunk)

            #rerank and get top N results
            reranked_results = co.rerank(model="rerank-v3.5", query = query, documents = docs, top_n = limit)
            
            # create dictionaries based of the best scoring sources from the reranker
            for item in reranked_results.results:
                id += 1
                logging.debug(f"Relevance score for source {id}: {item.relevance_score}")
                index = item.index
                chunk = results[index]
                
                chunk_dict = {
                    "id": id,
                    "source_url": chunk.source,
                    "chunk": chunk.chunk
                }
                json_results.append(chunk_dict)     
            return json_results
        except Exception as e:
            logging.error({"error during search_database": str(e)})
        finally:
            next(db_generator, None) # close the session.
            
        
    @property
    def hand_book_table(self):
        if self._handbook_table is None:
            try:
                self._handbook_table = self.dbmanager.get_table(settings.HANDBOOK_TABLE)
            except Exception as e:
                print(f"error accesssing table: {str(e)}")
                raise
        return self._handbook_table
    
    # @property
    # def history_table(self):
    #     if self._history_table is None:
    #         try:
    #             self._history_table = self.dbmanager.get_table(settings.HISTORY_TABLE)
    #         except Exception as e:
    #             print(f"error accesssing table: {str(e)}")
    #             raise
    #     return self._history_table
    
    # def get_chat_history(self, session_id: str):
    #     try:
    #         results = self.history_table.search().where(f"session_id = '{session_id}'").limit(1).to_pydantic(ChatHistory)
    #         if results:
    #             return results[0].history, results[0].share_token
    #         else:
    #             return None
    #     except Exception as e:
    #         logging.error(f'error retrieving chat history: {e}')
    
    # def update_chat_history(self, session_id: str, share_token: str, new_history: str):
    #     try:
    #         self.dbmanager.update_chat_history(table_name="history_table", session_id=session_id, share_token=share_token, new_history=new_history)
    #     except Exception as e:
    #         logging.error(f"something went wrong updating history : {e}")
       
    def get_tool_results(self, result: object, tool_name: str, history: List):
        content = []
        sources = []
        tools = []
        
        # get resuls from tool call out of Result object
        for message in result.all_messages():
                    for part in message.parts:
                        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name:
                            content.extend(part.content)
                            tools.append(part.tool_name)
                            system_call = {'role': "System", "content": f"Called tool: {tool_name}. Results: {part.content}"}
                            history.append(system_call)

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
    async def process_answer(self, history: str, session_id: str, share_token: str):
            response = None
            old_content = ""
            
            trace_id = langfuse_context.get_current_trace_id()
            logging.debug(langfuse_context.get_current_trace_url())
            
            model = get_model()        
            agent = Agent(model, result_type=ResponseModel, system_prompt= get_chatbot_prompt(self.language))    

          
            async with agent.run_stream(str(history)) as result:

                sources, tools_used = self.get_tool_results(result, 'search_database', history)
                
                metadata = ResponseMetadata(
                    sources = sources,
                    tools_used = tools_used,
                    session_id = session_id,
                    trace_id = trace_id,
                    share_token = share_token) 
                yield(json.dumps(metadata).encode('utf-8') + b'\n')      
                       
                async for structured_result, last in result.stream_structured(debounce_by=0.01): 
                        
                    try:
                        chunk = await result.validate_structured_result(
                        structured_result, allow_partial=not last
                        )                        
                        
                        content = chunk.get('content')
                        if not last:
                            if content != old_content:
                                old_content = content
                                # create response object
                                response = ResponseModel(
                                    content=chunk.get('content'), 
                                    )    
                                yield(json.dumps(response).encode('utf-8') + b'\n')   
                        else: 
                            
                            # create response object
                            response = ResponseModel(
                                content=content, 
                                able_to_answer=chunk.get("able_to_answer"),
                                question_classification= chunk.get('question_classification'),
                                follow_up_questions=chunk.get('follow_up_questions'))
                            yield(json.dumps(response).encode('utf-8') + b'\n')

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
                    
            history.append({'role': "Assistant", "content": content})
            #self.update_chat_history(session_id, share_token, json.dumps(history))

    async def generate_response_stream(self, request: RequestModel):
        # define variables
        history = []
        session_id = ""
        share_token = ""
        

        logging.debug(f"start {session_id}")
        # start langfuse trace
        trace = langfuse.trace(
            name = "chat_request"
        )


        # if session_id is null, generate one
        if not request.metadata['session_id']:
            logging.debug("test")
            session_id = str(uuid.uuid4())
            share_token = str(uuid.uuid4())
        # else:
        #     # if session_id exists, retrieve chat history
        #     history, share_token = self.get_chat_history(request.metadata['session_id'])
        #     session_id = request.metadata['session_id']
        #     history = json.loads(history)
        
        # add user question to history
        history.append({'role': "User", "content": request.user['question']})
        
        # get streaming response from agent
        retries = 1
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
        
    def generate_response(self, request: RequestModel):
        
        # if session_id is null, generate one
        if not request.metadata['session_id']:
            request.metadata['session_id'] = str(uuid.uuid4())
            share_token = str(uuid.uuid4())
        else:
            # if session_id exists, retrieve chat history
            history, share_token = self.get_chat_history(request.metadata['session_id'])
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
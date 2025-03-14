from __future__ import annotations as _annotations

from fastapi import APIRouter



from fastapi import APIRouter
from models.SQL_models import *
from models.models import SearchRequest
from components.DatabaseManager import get_session
from config import settings
import cohere
import logging
from sqlalchemy.orm import declarative_base, sessionmaker, Session, scoped_session
import openai
import numpy as np


router = APIRouter()

@router.post("/search/")
def search_database(request: SearchRequest):
        """
        Searches the database using the provided query for relevant chunks of information.
        
        This method performs a vector search on the PostgreSQL database containing embedded chunks of the Gitlab Handbook.
        It returns the top X matching results from the database in the form of a JSON list of dictionaries with ID, source and content
        Args:
            ctx (RunContext): The context of the current run, providing access to dependencies and state.
            query (str): The search query to use against the database.
            tool_call_attempt (int): The attempt number of the tool call, used to generate unique IDs for results. First attempt is 0.

        Returns:
            list[dict] or dict: A list of dictionaries, where each dictionary represents a search result
                            containing the 'id', 'source_url', and 'chunk'. 
        """ 
        #setup
        query = request.query
        tool_call_attempt = request.tool_call_attempt
        limit = request.limit
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
                    "chunk": chunk.chunk,
                    "relevance_score": item.relevance_score
                }
                json_results.append(chunk_dict)     
            return json_results
        except Exception as e:
            logging.error({"error during search_database": str(e)})
        finally:
            next(db_generator, None) # close the session.
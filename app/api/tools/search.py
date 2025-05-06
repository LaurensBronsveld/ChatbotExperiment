from __future__ import annotations as _annotations

from fastapi import APIRouter



from fastapi import APIRouter
from app.models.SQL_models import *
from app.models.models import SearchRequest
from app.components.DatabaseManager import get_session
from app.config import settings
import cohere
import logging
from sqlalchemy.orm import declarative_base, sessionmaker, Session, scoped_session, joinedload
from sqlalchemy import desc
import openai
import numpy as np


router = APIRouter()

def dense_search(
    query: str,
    db: scoped_session,
    query_method: str = "cosine_distance",
    n: int = 10
) -> list[dict]:
    """
    Performs dense vector similarity search using OpenAI embeddings and pgvector.

    Args:
        query (str): The natural language query to embed and search with.
        db (scoped_session): SQLAlchemy database session.
        query_method (str): Distance metric to use for similarity (e.g., cosine_distance).
        n (int): Number of top results to retrieve.
        explain (bool): Whether to return additional debug information.

    Returns:
        list[dict]: A list of chunks with their similarity scores.
    """
    try:
        print('test')
        # Get vector embedding for the query
        query_embedding = openai.embeddings.create(
            input=query,
            model=settings.EMBED_MODEL
        ).data[0].embedding
        query_vector = np.array(query_embedding).tolist()

        # Distance function map
        comparators = {
            "l2_distance": Chunk.embedding.l2_distance(query_vector),
            "l1_distance": Chunk.embedding.l1_distance(query_vector),
            "cosine_distance": Chunk.embedding.cosine_distance(query_vector),
            "dot_product": Chunk.embedding.max_inner_product(query_vector)
        }

        if query_method not in comparators:
            raise ValueError(f"Invalid query method '{query_method}'")

        comparator = comparators[query_method]

        # Run DB query with eager-loaded document
        query_results = (
            db.query(Chunk, comparator.label("score"))
            # .options(joinedload(Chunk.document))
            .order_by(comparator)
            .limit(n)
            .all()
        )

        # Normalize cosine distance (lower = better)
        scored_chunks = [
            {
                "chunk": chunk,
                "similarity_score": 1 - score if query_method == "cosine_distance" else score
            }
            for chunk, score in query_results
        ]

        return scored_chunks
    except Exception as e:
        print(f"[sparse_search] Search failed: {e}")
        return []


def sparse_search(query: str, db: scoped_session, n: int = 10) -> list[dict]:
    """
    Performs sparse full-text search using PostgreSQL's tsvector and ts_rank_cd.

    Args:
        query (str): The user query for text search.
        db (scoped_session): SQLAlchemy session.
        n (int): Number of results to return.

    Returns:
        list[dict]: Ranked chunks with relevance scores.
    """
    try:
        tsquery = func.websearch_to_tsquery('english', query)
        rank_func = func.ts_rank_cd(Chunk.chunk_tsv, tsquery).label('rank')

        results = (
            db.query(Chunk, rank_func)
            # .options(joinedload(Chunk.document))
            .where(Chunk.chunk_tsv.op('@@')(tsquery))
            .order_by(desc(rank_func))
            .limit(n)
            .all()
        )

        return [
            {"chunk": chunk, "rank": rank}
            for chunk, rank in results
        ]
    except Exception as e:
        print(f"[sparse_search] Search failed: {e}")
        return []

  
    
def hybrid_search(query: str, db: scoped_session, dense_comparator: str = "cosine_distance", alpha: float = 0.5, n: int = 10):
    """
    Performs a hybrid search combining dense and sparse retrieval methods.
    
    :param query: Search query
    :param dense_comparator: Similarity metric for dense search
    :param alpha: Weight for combining scores (0.5 means equal weight to both)
    :param n: Number of top results to return
    :return: List of combined search results
    """
    
    # Perform both searches
    dense_results = dense_search(query, db, dense_comparator, n)
    sparse_results = sparse_search(query, db, n)

    # Create a dictionary to store combined scores
    result_dict = {}
    results = []

    # Normalize Dense Scores
    max_dense_score = max(item["similarity_score"] for item in dense_results) if dense_results else 1
    for item in dense_results:
        norm_dense_score = item["similarity_score"] / max_dense_score
        result_dict[item["chunk"].id] = {"chunk": item["chunk"], "dense_score": norm_dense_score, "sparse_score": 0}

    # Normalize Sparse Scores
    max_sparse_score = max(item["rank"] for item in sparse_results) if sparse_results else 1
    for item in sparse_results:
        norm_sparse_score = item["rank"] / max_sparse_score
        if item["chunk"].id in result_dict:
            result_dict[item["chunk"].id]["sparse_score"] = norm_sparse_score
        else:
            result_dict[item["chunk"].id] = {"chunk": item["chunk"], "dense_score": 0, "sparse_score": norm_sparse_score}

    # Compute Hybrid Score
    for chunk_id, values in result_dict.items():
        values["hybrid_score"] = alpha * values["dense_score"] + (1 - alpha) * values["sparse_score"]
        results.append({
            "chunk": values["chunk"],
            "hybrid_score": values["hybrid_score"],
            "dense_score": values["dense_score"],
            "sparse_score": values["sparse_score"]      
        })

    # Sort results by hybrid score (descending)
    sorted_results = sorted(results, key=lambda x: x["hybrid_score"], reverse=True)
    
    # Return top-N results
    return sorted_results
 

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
            logging.debug(f"Query: {query}")
            #initialise Cohere reranker
            co = cohere.Client(settings.COHERE_API_KEY)
           
            db_generator = get_session()
            db = next(db_generator)

            # perform hybrid search on database
            results = hybrid_search(query, db)
            for result in results:
                print(result)
                break
            chunks = [result["chunk"] for result in results]
      
            #get list of strings to rerank with Cohere
            for chunk in chunks:
                docs.append(chunk.chunk)

            #rerank and get top N results
            reranked_results = co.rerank(model="rerank-v3.5", query = query, documents = docs, top_n = limit)
            
            # create dictionaries based of the best scoring sources from the reranker
            for item in reranked_results.results:
                id += 1

                index = item.index
                chunk = chunks[index]
                logging.debug(f"Relevance score for source {id} with chunk id {chunk.id} and title {chunk.document.title}: {item.relevance_score}")
                
                chunk_dict = {
                    "id": id,
                    "source_url": chunk.document.location,
                    "chunk": chunk.chunk,
                    "relevance_score": item.relevance_score
                }
                json_results.append(chunk_dict)     
            return json_results
        except Exception as e:
            logging.error({"error during search_database": str(e)})
        finally:
            db.close()

def main():
    dense_search("test", next(get_session()))


if __name__ == "__main__":
    

    main()

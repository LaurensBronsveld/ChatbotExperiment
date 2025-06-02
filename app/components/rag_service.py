from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

import cohere
import numpy as np
from httpx import HTTPError
from openai import OpenAI, OpenAIError
from sqlalchemy import desc, select
from sqlalchemy.orm import scoped_session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import (
    selectinload,
)
from sqlalchemy.sql import func

from app.core.config import settings
import logging as logger
from app.models.SQL_models import Chunk
from app.models.models import ChunkModel

if TYPE_CHECKING:
    from sqlalchemy.sql import ColumnElement


class RagService:
    """
    Encapsulates dense, sparse, and hybrid search logic over embedded chunks.
    """

    def __init__(
        self,
        db: scoped_session,
        openai_client: OpenAI | None = None,
        cohere_client: cohere.Client | None = None,
        embed_model: str = settings.EMBED_MODEL,
        rerank_model: str = settings.COHERE_RERANK_MODEL,
    ) -> None:
        """
        Initialize the RAG service with database session and optional API clients.

        Args:
            db: SQLAlchemy database session
            openai_client: Optional pre-configured OpenAI client
            cohere_client: Optional pre-configured Cohere client
            embed_model: Model name to use for embeddings
            rerank_model: Model name to use for reranking
        """
        self.db = db
        self.openai_client = openai_client or OpenAI(api_key=settings.OPENAI_API_KEY)
        self.cohere_client = cohere_client or cohere.Client(settings.COHERE_API_KEY)
        self.embed_model = embed_model
        self.rerank_model = rerank_model

    def _get_query_embedding(self, query: str) -> list[Any]:
        """
        Get vector embedding for a query text.

        Args:
            query: Text to be embedded

        Returns:
            List of embedding values as floats

        """
        try:
            query_embedding = (
                self.openai_client.embeddings.create(
                    input=query,
                    model=self.embed_model,
                    dimensions=settings.EMBEDDING_DIMENSION,
                )
                .data[0]
                .embedding
            )
            return np.array(query_embedding).tolist()
        except OpenAIError as e:
            logger.exception(f"[_get_query_embedding] Error getting embedding: {e}")
            raise

    def _get_distance_function(
        self, comparator: str, query_vector: list[float]
    ) -> ColumnElement:
        """
        Build a SQLAlchemy-compatible distance function for pgvector comparisons.

        Args:
            comparator: The distance metric to use.
            query_vector: The vector to compare against.

        Returns:
            SQLAlchemy expression for the selected distance function.
        """
        if comparator == "l2_distance":
            return Chunk.dense_embedding.l2_distance(query_vector)
        if comparator == "l1_distance":
            return Chunk.dense_embedding.l1_distance(query_vector)
        if comparator == "cosine_distance":
            return Chunk.dense_embedding.cosine_distance(query_vector)
        if comparator == "dot_product":
            return Chunk.dense_embedding.max_inner_product(query_vector)

        logger.warning(f"Chosen comparator: {comparator} does not exist.")
        logger.warning("Using default Cosine Distance comparator instead")
        return Chunk.dense_embedding.cosine_distance(query_vector)

    def dense_search(
        self,
        query: str,
        comparator: str = "cosine_distance",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Performs dense vector similarity search using OpenAI embeddings and pgvector.

        Args:
            query: The natural language query to embed and search with.
            comparator: Distance metric to use for similarity
                        ("cosine_distance", "l2_distance", "l1_distance", "dot_product")
            top_k: Number of top results to retrieve.

        Returns:
            List of chunks with their similarity scores.
        """
        # Get vector embedding for the query
        query_vector = self._get_query_embedding(query)
        try:
            # Get distance function for SQL query
            distance_func = self._get_distance_function(comparator, query_vector)

            # Run DB query
            statement = (
                select(Chunk, distance_func.label("score"))
                .options(selectinload(Chunk.document))
                .order_by(distance_func)
                .limit(top_k)
            )
            query_results = self.db.execute(statement)

            # combine chunk with score
            scored_chunks = [
                {
                    "chunk": chunk,
                    "score": 1 - score
                    if comparator == "cosine_distance"
                    else score,  # change cosine distance into cosine similarity
                }
                for chunk, score in query_results
            ]
            logger.info(f"retrieved dense chunks: {len(scored_chunks)}")

        except SQLAlchemyError as e:
            logger.exception(f"[dense_search] Search failed: {e}")
            return []
        else:
            return scored_chunks

    def sparse_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Performs sparse full-text search using PostgreSQL's tsvector and ts_rank_cd.

        Args:
            query: The user query for text search.
            top_k: Number of results to return.

        Returns:
            List of ranked chunks with relevance scores.
        """
        try:
            tsquery = func.websearch_to_tsquery("dutch", query)
            rank_func = func.ts_rank_cd(Chunk.sparse_embedding, tsquery).label("rank")

            statement = (
                select(Chunk, rank_func)
                .options(selectinload(Chunk.document))
                .where(Chunk.sparse_embedding.op("@@")(tsquery))
                .order_by(desc(rank_func))
                .limit(top_k)
            )
            query_results = self.db.execute(statement)
            # combine chunk with score
            scored_chunks = [
                {"chunk": chunk, "score": score} for chunk, score in query_results
            ]
            logger.info(f"retrieved sparse chunks: {len(scored_chunks)}")

        except SQLAlchemyError as e:
            logger.exception(f"[sparse_search] Search failed: {e}")
            return []
        else:
            return scored_chunks

    def _normalize_scores(self, items: list[dict[str, Any]]) -> dict[int, float]:
        """
        Normalize scores from a list of items.

        Args:
            items: List of dictionaries containing chunks and their scores

        Returns:
            Dictionary mapping chunk IDs to normalized scores
        """
        if not items:
            return {}

        max_score = items[0]["score"]

        if max_score == 0:
            return {item["chunk"].id: 0 for item in items}

        return {item["chunk"].id: item["score"] / max_score for item in items}

    def hybrid_search(
        self,
        query: str,
        dense_comparator: str = "cosine_distance",
        alpha: float = 0.5,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Performs a hybrid search combining dense and sparse retrieval methods.

        Args:
            query: Search query
            dense_comparator: Similarity metric for dense search
            alpha: Weight for combining scores  (1: dense, 0: sparse)
            top_k: Number of top results to return

        Returns:
            List of combined search results with hybrid scores
        """
        logger.info(f"Searching database with query: {query}")
        # Perform both searches
        dense_results = self.dense_search(query, dense_comparator, top_k)
        sparse_results = self.sparse_search(query, top_k)

        # Normalize scores
        norm_dense_scores = self._normalize_scores(dense_results)
        norm_sparse_scores = self._normalize_scores(sparse_results)

        # Create a dictionary to store combined scores
        result_dict: dict[int, dict[str, Any]] = {}

        # Process dense results
        for item in dense_results:
            chunk_id = item["chunk"].id
            result_dict[chunk_id] = {
                "chunk": item["chunk"],
                "dense_score": norm_dense_scores.get(chunk_id, 0),
                "sparse_score": 0,
            }

        # Process sparse results
        for item in sparse_results:
            chunk_id = item["chunk"].id
            if chunk_id in result_dict:
                result_dict[chunk_id]["sparse_score"] = norm_sparse_scores.get(
                    chunk_id, 0
                )
            else:
                result_dict[chunk_id] = {
                    "chunk": item["chunk"],
                    "dense_score": 0,
                    "sparse_score": norm_sparse_scores.get(chunk_id, 0),
                }

        # Compute hybrid scores and prepare output
        results = []
        for values in result_dict.values():
            hybrid_score = (
                alpha * values["dense_score"] + (1 - alpha) * values["sparse_score"]
            )
            results.append(
                {
                    "chunk": values["chunk"],
                    "hybrid_score": hybrid_score,
                    "dense_score": values["dense_score"],
                    "sparse_score": values["sparse_score"],
                }
            )

        # Sort results by hybrid score (descending)
        sorted_results = sorted(results, key=lambda x: x["hybrid_score"], reverse=True)

        # Return top-K results
        return sorted_results[:top_k]

    def rerank_and_format(
        self,
        chunks: list[Chunk],
        query: str,
        tool_call_attempt: int,
        limit: int,
    ) -> list[ChunkModel]:
        """
        Use Cohere to rerank the chunks and build the final response payload.

        Args:
            chunks: List of database chunk objects
            query: Original search query
            tool_call_attempt: Current attempt number
            limit: Maximum number of results to return

        Returns:
            List of formatted chunk schemas with relevance scores
        """
        try:
            docs = [c.text for c in chunks]
            reranked = self.cohere_client.rerank(
                model=self.rerank_model,
                query=query,
                documents=docs,
                top_n=limit,
            ).results

            base_id = tool_call_attempt * limit
            output: list[ChunkModel] = []

            for item in reranked:
                base_id += 1
                chunk = chunks[item.index]

                output.append(
                    ChunkModel(
                        id=base_id,
                        chunk_id=chunk.id,
                        document_title=chunk.document.title,
                        document_location=chunk.document.location,
                        text=chunk.text,
                        relevance_score=item.relevance_score,
                    )
                )
        except HTTPError as e:
            logger.error(f"[rerank_and_format] cohere Reranking failed: {e}")
            return []
        else:
            return output

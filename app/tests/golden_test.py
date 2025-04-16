import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import csv
import json
from typing import List, Dict, Tuple, Any, Optional
import os
from uuid import uuid4

from app.components.DatabaseManager import get_session
from app.api.tools.search import search_database
from app.agents.BaseAgent import BaseAgent
from app.models.models import *
from app.tests.example_requests import *

class RAGEvaluator:
    def __init__(self, golden_test_set_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG evaluation system
        
        Args:
            golden_test_set_path: Path to the CSV file containing the golden test set
            model_name: The sentence transformer model to use for semantic similarity
        """
        self.golden_test_set_path = golden_test_set_path
        self.golden_df = self._load_golden_test_set()
        self.encoder = SentenceTransformer(model_name)
        
    def _load_golden_test_set(self) -> pd.DataFrame:
        """Load the golden test set from CSV"""
        try:
            # Try reading with common encodings
            for encoding in ['utf-8', 'latin-1', 'ISO-8859-1']:
                try:
                    df = pd.read_csv(self.golden_test_set_path, encoding=encoding, delimiter=';')
                    return df
                except UnicodeDecodeError:
                    continue
                    
            # If all encodings fail, try excel format
            return pd.read_excel(self.golden_test_set_path)
            
        except Exception as e:
            print(f"Error loading golden test set: {e}")
            # Create empty DataFrame with expected columns if loading fails
            return pd.DataFrame(columns=["ID", "Question", "Answer", "Source File", "Relevant Section"])
    
    def _normalize_path(self, path: str) -> str:
        """Normalize file paths for comparison"""
        # Remove leading/trailing whitespace, normalize slashes
        path = path.strip()
        path = path.replace('\\', '/')
        
        # Remove leading "/" if present
        if path.startswith('/'):
            path = path[1:]
            
        # Handle cases where the path might be in a different format
        path = re.sub(r'^.*?content/', 'content/', path)
        
        return path.lower()
    
    def evaluate_retrieval(self, question_id: int, retrieved_sources: List[str]) -> Dict[str, Any]:
        """
        Evaluate if the retrieved sources match the expected sources for a question
        
        Args:
            question_id: The ID of the question to evaluate
            retrieved_sources: List of source paths retrieved by the RAG system
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get the golden source(s) for this question
        question_row = self.golden_df[self.golden_df["ID"] == question_id]
        if question_row.empty:
            return {"error": f"Question ID {question_id} not found in golden test set"}
        
        golden_sources_raw = question_row["Source File"].iloc[0]
        golden_sources = [self._normalize_path(src.strip()) for src in golden_sources_raw.split(',')]
        
        # Normalize retrieved sources
        normalized_retrieved = [self._normalize_path(src['url']) for src in retrieved_sources]
        
        # Check exact matches
        exact_matches = set(golden_sources).intersection(set(normalized_retrieved))
        print(f"golden sources: {golden_sources}")
        print(f"retrieved sources: {normalized_retrieved}")

        # Calculate partial matches (if a retrieved source contains or is contained in a golden source)
        partial_matches = []
        for g_src in golden_sources:
            for r_src in normalized_retrieved:
                if g_src in r_src or r_src in g_src:
                    if (g_src, r_src) not in partial_matches and (r_src, g_src) not in partial_matches:
                        partial_matches.append((g_src, r_src))
        
        # Calculate metrics
        precision = len(exact_matches) / len(normalized_retrieved) if normalized_retrieved else 0
        recall = len(exact_matches) / len(golden_sources) if golden_sources else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "question_id": question_id,
            "golden_sources": golden_sources,
            "retrieved_sources": normalized_retrieved,
            "exact_matches": list(exact_matches),
            "partial_matches": partial_matches,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "retrieval_success": len(exact_matches) > 0 or len(partial_matches) > 0
        }
    
    def evaluate_answer(self, question_id: int, generated_answer: str) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated answer against the golden answer
        
        Args:
            question_id: The ID of the question to evaluate
            generated_answer: The answer generated by the RAG system
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get the golden answer for this question
        question_row = self.golden_df[self.golden_df["ID"] == question_id]
        if question_row.empty:
            return {"error": f"Question ID {question_id} not found in golden test set"}
        
        golden_answer = question_row["Answer"].iloc[0]
        question = question_row["Question"].iloc[0]
        
        # Calculate semantic similarity
        golden_embedding = self.encoder.encode([golden_answer])[0]
        generated_embedding = self.encoder.encode([generated_answer])[0]
        semantic_similarity = cosine_similarity([golden_embedding], [generated_embedding])[0][0]
        
        # Simple lexical overlap (Jaccard similarity)
        golden_tokens = set(re.findall(r'\b\w+\b', golden_answer.lower()))
        generated_tokens = set(re.findall(r'\b\w+\b', generated_answer.lower()))
        
        jaccard = len(golden_tokens.intersection(generated_tokens)) / len(golden_tokens.union(generated_tokens)) if golden_tokens or generated_tokens else 0
        
        return {
            "question_id": question_id,
            "question": question,
            "golden_answer": golden_answer,
            "generated_answer": generated_answer,
            "semantic_similarity": float(semantic_similarity),
            "lexical_overlap": jaccard
        }
    
    def run_evaluation(self, rag_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a full evaluation on a batch of RAG results
        
        Args:
            rag_results: List of dictionaries, each containing:
                - question_id: The ID of the question
                - retrieved_sources: List of source paths retrieved by the RAG
                - generated_answer: The answer generated by the system
                
        Returns:
            Dictionary with aggregated evaluation metrics
        """
        retrieval_results = []
        answer_results = []
        
        for result in rag_results:
            question_id = result["question_id"]
            retrieved_sources = result.get("retrieved_sources", [])
            generated_answer = result.get("generated_answer", "")
            
            retrieval_eval = self.evaluate_retrieval(question_id, retrieved_sources)
            retrieval_results.append(retrieval_eval)
            
            answer_eval = self.evaluate_answer(question_id, generated_answer)
            answer_results.append(answer_eval)
        
        # Calculate aggregate metrics
        retrieval_success_rate = sum(1 for r in retrieval_results if r.get("retrieval_success", False)) / len(retrieval_results) if retrieval_results else 0
        avg_retrieval_precision = sum(r.get("precision", 0) for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0
        avg_retrieval_recall = sum(r.get("recall", 0) for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0
        avg_retrieval_f1 = sum(r.get("f1_score", 0) for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0
        
        avg_semantic_similarity = sum(a.get("semantic_similarity", 0) for a in answer_results) / len(answer_results) if answer_results else 0
        avg_lexical_overlap = sum(a.get("lexical_overlap", 0) for a in answer_results) / len(answer_results) if answer_results else 0
        
        return {
            "num_questions_evaluated": len(rag_results),
            "retrieval_metrics": {
                "success_rate": retrieval_success_rate,
                "avg_precision": avg_retrieval_precision,
                "avg_recall": avg_retrieval_recall,
                "avg_f1": avg_retrieval_f1
            },
            "answer_metrics": {
                "avg_semantic_similarity": float(avg_semantic_similarity),
                "avg_lexical_overlap": avg_lexical_overlap
            },
            "detailed_results": {
                "retrieval": retrieval_results,
                "answers": answer_results
            }
        }
    
    def save_results(self, evaluation_results: Dict[str, Any], output_path: str = "rag_evaluation_results.json"):
        """Save evaluation results to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"Evaluation results saved to {output_path}")
        
        # Also save a CSV summary of per-question results
        with open(output_path.replace('.json', '_summary.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Question ID", "Question", "Retrieval Success", "Precision", "Recall", 
                "F1 Score", "Semantic Similarity", "Lexical Overlap"
            ])
            
            retrieval_results = evaluation_results["detailed_results"]["retrieval"]
            answer_results = evaluation_results["detailed_results"]["answers"]
            
            for i in range(len(retrieval_results)):
                r = retrieval_results[i]
                a = answer_results[i]
                writer.writerow([
                    r.get("question_id", "N/A"),
                    a.get("question", "N/A"),
                    r.get("retrieval_success", False),
                    r.get("precision", 0),
                    r.get("recall", 0),
                    r.get("f1_score", 0),
                    a.get("semantic_similarity", 0),
                    a.get("lexical_overlap", 0)
                ])
        print(f"Summary CSV saved to {output_path.replace('.json', '_summary.csv')}")

# Example usage function to demonstrate how to use the evaluator with an existing RAG system
def evaluate_rag_system(golden_test_set_path: str, llm_function):
    """
    Evaluate a RAG system against the golden test set
    
    Args:
        golden_test_set_path: Path to the CSV file containing the golden test set
        rag_function: Function that takes a question and returns retrieved sources
        llm_function: Function that takes a question and retrieved chunks and returns an answer
    """
    evaluator = RAGEvaluator(golden_test_set_path)
    golden_df = evaluator.golden_df
    
    rag_results = []
    
    # Process each question in the golden test set
    for _, row in golden_df.iterrows():
        question_id = row["ID"]
        question = row["Question"]
        
        print(f"Evaluating question {question_id}: {question[:50]}...")
        
        # # Call your existing RAG retrieval function
        # retrieved_chunks = rag_function(question)
        
        # # Extract just the source paths from the retrieved chunks
        # # Adjust this based on how your system returns sources
        # retrieved_sources = [chunk["source"] for chunk in retrieved_chunks]
        
        # Call your existing LLM answer generation function
        llm_response = llm_function(question)
        sources = llm_response['sources']
        llm_answer = llm_response['answer']
        
        rag_results.append({
            "question_id": question_id,
            "retrieved_sources": sources,
            "generated_answer": llm_answer
        })
    
    # Run the evaluation
    evaluation_results = evaluator.run_evaluation(rag_results)
    
    # Save the results
    evaluator.save_results(evaluation_results)
    
    # Print summary metrics
    print("\nEvaluation Summary:")
    print(f"Number of questions evaluated: {evaluation_results['num_questions_evaluated']}")
    print(f"Retrieval success rate: {evaluation_results['retrieval_metrics']['success_rate']:.2f}")
    print(f"Average retrieval F1 score: {evaluation_results['retrieval_metrics']['avg_f1']:.2f}")
    print(f"Average answer semantic similarity: {evaluation_results['answer_metrics']['avg_semantic_similarity']:.2f}")
    
    return evaluation_results

# Example of how to integrate with your existing RAG system
if __name__ == "__main__":
    # Replace these functions with your actual implementations
    db = next(get_session())
    
    GOLDEN_TEST_PATH = "data/gitlab-handbook-golden-test-set.csv"
    # def my_rag_search(question):
    #     # This should call your existing RAG search function
    #     # Return format should be a list of dictionaries, each with at least a "source" field
    #     # Example: [{"source": "/content/handbook/values/index.md", "content": "..."}]
        
    #     results = search_database(question, db)
    #     chunks = [result['chunk'] for result in results]
    #     chunks_dict = [{"source": chunk.document.location, "content": chunk.chunk} for chunk in chunks]
    #     for dict in chunks_dict:
    #         print(dict['source'])
    #     return chunks_dict
        
    
    def my_llm_answer(question):
        # This should call your existing LLM answer generation function
        # It should return the generated answer as a string
        assistant = BaseAgent()
        request = get_request(question)
        session_id = uuid4()
        response = assistant.generate_simple_response(question)
        return response

    
    # Run the evaluation
    evaluation_results = evaluate_rag_system(GOLDEN_TEST_PATH, my_llm_answer)
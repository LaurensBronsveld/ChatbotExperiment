import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.main import app
from app.tests.example_requests import *
from app.models.models import *
import json
from httpx import AsyncClient
import pytest
import logging
from app.config import settings
from app.agents.LLMs import get_model
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from app.agents.Judge import JudgeAgent, JudgeInput
from app.agents.BaseAgent import BaseAgent
from app.evaluation.rag_evaluator import RagEvaluator

API_URL = "/api"


client = TestClient(app)

def evaluate_RAG_response(golden_testset_path: str, output_path: str):
    """

    """
    # GOLDEN_TEST_PATH = "data/gitlab_rag_golden_testset.csv"
    rag_evaluator = RagEvaluator(golden_testset_path)
    judge_agent = JudgeAgent()
    assistant = BaseAgent()
    
    df = pd.read_csv(GOLDEN_TEST_PATH, delimiter=',')
    rag_results = []

    # Process each question in the golden test set
    for _, row in df.iterrows():
        question_id = row["id"]
        question = row["question"]
        
        print(f"Evaluating question {question_id}: {question}...")
        
        try:
        # get chat response
            response = assistant.generate_simple_response(question)
        except Exception as e:
            print(e)
            return
        
        answer = response["answer"]
        source_dicts = response['sources']
        sources = [source_dict['url'] for source_dict in source_dicts]
        texts = [source_dict['text'] for source_dict in source_dicts]
        
        rag_results.append({
            "question_id": question_id,
            "retrieved_sources": sources,
            "retrieved_texts": texts,
            "generated_answer": answer
        })
        
     # Run the evaluation
    evaluation_results = rag_evaluator.run_evaluation(rag_results)
    
    # Save the results
    rag_evaluator.save_results(output_path, evaluation_results)
    
    # Print summary metrics
    print("\nEvaluation Summary:")
    print(f"Number of questions evaluated: {evaluation_results['num_questions_evaluated']}")
    print(f"Average retrieval F1 score: {evaluation_results['retrieval_metrics']['avg_f1']:.2f}")
    print(f"Average score determined by judge: {evaluation_results['answer_metrics']['avg_judge_score']}")
    return evaluation_results


if __name__ == "__main__":
    
    #GOLDEN_TEST_PATH = "data/gitlab-handbook-golden-test-set.csv"
    GOLDEN_TEST_PATH = "data/gitlab_rag_golden_testset.csv"
    RESULT_PATH = "data/gitlab_golden_testset_evaluation.csv"
    evaluate_RAG_response(GOLDEN_TEST_PATH, RESULT_PATH)

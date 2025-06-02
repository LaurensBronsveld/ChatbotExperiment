import pandas as pd
from app.tests.example_requests import *
from app.models.models import *


from app.agents.BaseAgent import BaseAgent
from app.evaluation.rag_evaluator import RagEvaluator

API_URL = "/api"


def evaluate_RAG_response(golden_testset_path: str, output_path: str):
    """
    Evaluates the RAG system's responses against a golden test set.

    This function reads a golden test set, generates answers for each question
    using the BaseAgent, evaluates these answers and the retrieved sources using
    the RagEvaluator, saves the detailed results, and prints a summary
    of the evaluation metrics.

    Args:
        golden_testset_path (str): The file path to the golden test set (CSV format).
        output_path (str): The file path where the evaluation results (CSV format) will be saved.

    Returns:
        dict: A dictionary containing the aggregated evaluation metrics and detailed results.
              Returns None if an exception occurs during response generation early in the process.
    """
    rag_evaluator = RagEvaluator(golden_testset_path)
    assistant = BaseAgent()

    df = pd.read_csv(GOLDEN_TEST_PATH, delimiter=",")
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
        source_dicts = response["sources"]
        sources = [source_dict["url"] for source_dict in source_dicts]
        texts = [source_dict["text"] for source_dict in source_dicts]

        rag_results.append(
            {
                "question_id": question_id,
                "retrieved_sources": sources,
                "retrieved_texts": texts,
                "generated_answer": answer,
            }
        )

    # Run the evaluation
    evaluation_results = rag_evaluator.run_evaluation(rag_results)

    # Save the results
    rag_evaluator.save_results(output_path, evaluation_results)

    # Print summary metrics
    print("\nEvaluation Summary:")
    print(
        f"Number of questions evaluated: {evaluation_results['num_questions_evaluated']}"
    )
    print(
        f"Average retrieval F1 score: {evaluation_results['retrieval_metrics']['avg_f1']:.2f}"
    )
    print(
        f"Average score determined by judge: {evaluation_results['answer_metrics']['avg_judge_score']}"
    )
    return evaluation_results


if __name__ == "__main__":
    GOLDEN_TEST_PATH = "data/gitlab_rag_golden_testset.csv"
    RESULT_PATH = "data/gitlab_golden_testset_evaluation.csv"
    evaluate_RAG_response(GOLDEN_TEST_PATH, RESULT_PATH)

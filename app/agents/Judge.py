from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from app.agents.system_prompts import get_judge_prompt2


class JudgeInput(BaseModel):
    question: str = Field(description="Question to evaluate")
    golden_answer: str = Field(
        description="Ideal answer to the question to compare the actual answer with"
    )
    llm_answer: str = Field(description="Actual answer the LLM generated to evaluate.")
    source_documents: list[str]


class JudgeOutput(BaseModel):
    score: float = Field(description="Final score of the generated answer between 0-10")
    summary: str = Field(
        description="Brief evaluation of the generated answer and explanation of how you decided on the score"
    )


# Evaluation criteria models
class CriterionScore(BaseModel):
    score: float = Field(..., ge=0, le=10, description="Score from 0-10")
    explanation: str = Field(..., description="Brief explanation for the score")


class ResponseEvaluation(BaseModel):
    factual_accuracy: CriterionScore = Field(
        ..., description="Measures correctness of information"
    )
    completeness: CriterionScore = Field(
        ..., description="Measures if all aspects of the question are addressed"
    )
    relevance: CriterionScore = Field(
        ..., description="Measures if information is directly relevant"
    )
    clarity: CriterionScore = Field(
        ..., description="Measures if the response is clear and understandable"
    )
    conciseness: CriterionScore = Field(
        ..., description="Measures if the response is appropriately concise"
    )
    total_score: float = Field(..., ge=0, le=50, description="Sum of all scores")
    evaluation_summary: str = Field(..., description="Brief overall assessment")


class JudgeAgent:
    agent_name = "judge"
    default_model = "gpt-4o-mini"

    def __init__(self, model="gpt-4o-mini"):
        self.model = model

    def initialize_agent(self):
        model = OpenAIModel(self.default_model)
        agent = Agent(
            model=model, result_type=JudgeOutput, system_prompt=get_judge_prompt2()
        )
        return agent

    def evaluate_response(self, JudgeInput) -> ResponseEvaluation:
        agent = self.initialize_agent()
        query = f"""
            Evaluate this response against the golden answer:
            Question: {JudgeInput.question}
            Golden answer: {JudgeInput.golden_answer}
            Actual answer: {JudgeInput.llm_answer}
            """
        result = agent.run_sync(query)
        return result

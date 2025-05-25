from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from app.agents.system_prompts import get_judge_prompt2
from app.models.models import JudgeInput, JudgeOutput, ResponseEvaluation

class JudgeAgent:
    agent_name = "judge"
    default_model = "gpt-4o-mini"

    def __init__(self, model="gpt-4o-mini"):
        """
        Initializes the JudgeAgent.

        Args:
            model (str, optional): The model name to be used by the agent.
                                   Defaults to "gpt-4o-mini".
        """
        self.model = model

    def initialize_agent(self):
        """
        Initializes and returns the Pydantic AI Agent for the judge.

        Configures the agent with the specified model, a predefined system prompt,
        and the expected result type (JudgeOutput).

        Returns:
            Agent: An instance of the Pydantic AI Agent.
        """
        model = OpenAIModel(self.default_model)
        agent = Agent(
            model=model, result_type=JudgeOutput, system_prompt=get_judge_prompt2()
        )
        return agent

    def evaluate_response(self, judge_input: JudgeInput) -> ResponseEvaluation:
        """
        Evaluates an LLM's response against a golden answer using the initialized agent.

        Args:
            judge_input (JudgeInput): An object containing the question, golden_answer,
                                     and the llm_answer to be evaluated.

        Returns:
            ResponseEvaluation: The evaluation result from the agent, conforming to JudgeOutput.
                                This is synonymous with the JudgeOutput type.
        """
        agent = self.initialize_agent()
        query = f"""
            Evaluate this response against the golden answer:
            Question: {judge_input.question}
            Golden answer: {judge_input.golden_answer}
            Actual answer: {judge_input.llm_answer}
            """
        result = agent.run_sync(query)
        return result
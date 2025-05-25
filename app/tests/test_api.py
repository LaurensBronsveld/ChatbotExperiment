from fastapi.testclient import TestClient
from app.main import app
from app.tests.example_requests import *
from app.models.models import *

import json
from app.config import settings
from app.agents.LLMs import get_model
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.anthropic import AnthropicModel


API_URL = "/api"


client = TestClient(app)

def validate_response(endpoint: str, query: str):
    """
    Sends a POST request to the specified endpoint with the given query and validates the response.

    It checks if the HTTP status code is 200 and if the response metadata and content
    conform to the ResponseMetadata and ResponseModel Pydantic models respectively.

    Args:
        endpoint (str): The API endpoint URL to send the request to.
        query (str): The query string to be included in the request payload.

    Returns:
        tuple[dict, dict]: A tuple containing the parsed metadata (dict) and
                           the parsed response content (dict).

    Raises:
        AssertionError: If the status code is not 200, or if the response
                        metadata or content fail Pydantic model validation.
    """
    # get chat response
    result = client.post(endpoint, json=get_request(query))
    data = result.json()
    metadata = json.loads(data["metadata"])
    response = json.loads(data["response"])

    print(metadata)
    print(response)
    # assert succesful response
    assert result.status_code == 200

    # assert response follows expected models
    assert ResponseModel.model_validate(response)
    assert ResponseMetadata.model_validate(metadata)

    return metadata, response

def test_new_chat_without_RAG():
    """
    Tests if a new session is created if you post a simple question without session_id.
    Validates the response from the API if it correctly follows the predefined schemas ResponseModel and ResponseMetadata
    It should not use RAG.
    """
    endpoint = f"{API_URL}/chat_test/"

    metadata, response = validate_response(endpoint, "Wat kan je voor mij doen?")

    # assert it did not use RAG
    assert metadata["tools_used"] == []


def test_new_chat_response_with_RAG():
    """
    Tests if a new session is created if you post a question about the Gitlab Handbook without session_id.
    Validates the response from the API if it correctly follows the predefined schemas ResponseModel and ResponseMetadata
    Validates that it has used RAG to answer the question.
    """
    endpoint = f"{API_URL}/chat_test/"

    # get chat response
    metadata, response = validate_response(endpoint, "Hoe kan ik ziekte verlof aanvragen?")

    # assert it was able to answer the question
    assert response["able_to_answer"] == True
    global SESSION_ID
    SESSION_ID = metadata["session_id"]


def test_follow_up_question():
    """
    Tests if you can ask a follow up question in an already existing session.
    Validates the response from the API if it correctly follows the predefined schemas ResponseModel and ResponseMetadata
    Validates that it can read the history of the conversation and use it as context for answers.
    """
    endpoint = f"{API_URL}/chat_test/{SESSION_ID}/"

    metadata, response = validate_response(endpoint, "Herhaal mijn eerste vraag woord voor woord?")

    assert "Hoe kan ik ziekte verlof aanvragen?" in response["content"]


def test_changing_LLM_model():
    """
    Tests the ability to dynamically change the LLM provider and model via settings
    and validates that the correct model type is instantiated and used by the agent.
    Resets the LLM settings to default values after the test.
    """
    endpoint = f"{API_URL}/chat_test/"

    # change LLM model to gemini
    settings.LLM_PROVIDER = "google"
    settings.LLM_MODEL = "gemini-2.0-flash"

    # assert get_model now returns a gemini model which is used in the following test
    model = get_model()
    assert isinstance(model, GeminiModel)
    metadata, response = validate_response(endpoint, "Wat kan je voor mij doen?")

    # change LLM model to Claude
    settings.LLM_PROVIDER = "anthropic"
    settings.LLM_MODEL = "claude-3-7-sonnet-latest"

    # assert get_model now returns a gemini model which is used in the following test
    model = get_model()
    assert isinstance(model, AnthropicModel)
    metadata, response = validate_response(endpoint, "Wat kan je voor mij doen?")

    # reset provider and model
    settings.LLM_PROVIDER = "open-AI"
    settings.LLM_MODEL = "gpt-4o"


def test_streaming_endpoint():
    """
    Tests if the streaming endpoint correctly returns a streamed response
    using FastAPI's TestClient.stream() method.
    """
    endpoint = f"{API_URL}/chat/"

    # Use stream context manager
    with client.stream(
        "POST", endpoint, json=get_request("Wat kan je voor mij doen?")
    ) as response:
        # Check status code
        assert response.status_code == 200

        # Check headers for streaming response
        assert "application/json-stream" in response.headers.get(
            "content-type", ""
        ) or "text/event-stream" in response.headers.get("content-type", "")

        # Process and validate chunks
        chunks = []

        for chunk in response.iter_lines():
            if chunk:
                chunks.extend(chunk)

        # Verify we got multiple chunks
        assert len(chunks) > 1

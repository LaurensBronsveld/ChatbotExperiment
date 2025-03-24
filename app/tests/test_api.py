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


API_URL = "/api"



@app.get("/")
async def read_main():
    return {"msg": "Hello World"}


client = TestClient(app)


def test_new_chat_without_RAG():
    """
    Tests if a new session is created if you post a simple question without session_id.
    Validates the response from the API if it correctly follows the predefined schemas ResponseModel and ResponseMetadata
    It should not use RAG.
    """
    endpoint = f"{API_URL}/chat_test/"

    # get chat response
    result = client.post(endpoint, json = get_request("Wat kan je voor mij doen?"))
    data = result.json()
    metadata = json.loads(data['metadata'])
    response = json.loads(data['response'])
    

    print(metadata)
    print(response)
    # assert succesful response
    assert result.status_code == 200

    # assert response follows expected models
    assert ResponseModel.model_validate(response)
    assert ResponseMetadata.model_validate(metadata)

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
    result = client.post(endpoint, json = get_request("Hoe kan ik ziekte verlof aanvragen?"))
    data = result.json()
    metadata = json.loads(data['metadata'])
    response = json.loads(data['response'])
    

    print(metadata)
    print(response)
    # assert succesful response
    assert result.status_code == 200

    # assert response follows expected models
    assert ResponseModel.model_validate(response)
    assert ResponseMetadata.model_validate(metadata)

    # assert it was able to answer the question
    assert response["able_to_answer"] == True
    global SESSION_ID
    SESSION_ID = metadata['session_id']
    
    
def test_follow_up_question():
    """
    Tests if you can ask a follow up question in an already existing session.
    Validates the response from the API if it correctly follows the predefined schemas ResponseModel and ResponseMetadata
    Validates that it can read the history of the conversation and use it as context for answers.
    """
    endpoint = f"{API_URL}/chat_test/{SESSION_ID}/"

    # get chat response
    result = client.post(endpoint, json = get_request("Herhaal mijn eerste vraag woord voor woord?"))
    data = result.json()
    metadata = json.loads(data['metadata'])
    response = json.loads(data['response'])
    

    print(metadata)
    print(response)
    # assert succesful response
    assert result.status_code == 200

    # assert response follows expected models
    assert ResponseModel.model_validate(response)
    assert ResponseMetadata.model_validate(metadata)

    assert "Hoe kan ik ziekte verlof aanvragen?" in response["content"] 
    

def test_changing_LLM_model():
    
    # change LLM model to gemini
    settings.LLM_PROVIDER="google"
    settings.LLM_MODEL="gemini-2.0-flash"

    # assert get_model now returns a gemini model which is used in the following test
    model = get_model()
    assert isinstance(model, GeminiModel)
    test_new_chat_without_RAG()

    # change LLM model to Claude
    settings.LLM_PROVIDER="anthropic"
    settings.LLM_MODEL="claude-3-7-sonnet-latest"

    # assert get_model now returns a gemini model which is used in the following test
    model = get_model()
    assert isinstance(model, AnthropicModel)
    test_new_chat_without_RAG()
   
def test_streaming_endpoint():
    """
    Tests if the streaming endpoint correctly returns a streamed response
    using FastAPI's TestClient.stream() method.
    """
    endpoint = f"{API_URL}/chat/"
    
    # Use stream context manager
    with client.stream("POST", endpoint, json=get_request("Wat kan je voor mij doen?")) as response:
        # Check status code
        assert response.status_code == 200
        
        # Check headers for streaming response
        assert 'application/json-stream' in response.headers.get('content-type', '') or \
               'text/event-stream' in response.headers.get('content-type', '')


        # Process and validate chunks
        chunks = []
        for chunk in response.iter_lines():
            if chunk:
                chunks.extend(chunk)

                
        # Verify we got multiple chunks
        assert len(chunks) > 1 


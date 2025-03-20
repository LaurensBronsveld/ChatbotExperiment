from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.main import app
from app.tests.example_requests import first_request
from app.models.models import *
import json
from httpx import AsyncClient
import pytest



API_URL = "/api"


@app.get("/")
async def read_main():
    return {"msg": "Hello World"}


client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}

def test_chat_response():
    endpoint = f"{API_URL}/chat_test/"
    response = client.post(endpoint, json = first_request)

    assert response.status_code == 200
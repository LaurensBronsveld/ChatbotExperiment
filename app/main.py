from typing import Union
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import lancedb
import uvicorn
import pandas as pd


from agent import Assistant_Agent
from DatabaseManager import DatabaseManager
from models.chat import ResponseModel, QueryModel


uri = "./data/lancedb"
app = FastAPI()


db_manager = DatabaseManager(uri)
assistant = Assistant_Agent(db_manager)

async def generate_response_stream(query: str):
    async for chunk in assistant.generate_response("hello"):
        yield chunk

@app.post("/assistant/")
async def get_response(request: dict) -> ResponseModel:
    """
    Generate a response from the assistant agent.
    Returns a json with the answer and sources used.
    """

    query = request.get("query", "")
   
    # response = await assistant.generate_response(
    #     request
    #     )
    return StreamingResponse(generate_response_stream("hello"), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, log_level="info")

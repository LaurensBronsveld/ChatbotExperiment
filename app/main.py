from typing import Union
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import lancedb
import uvicorn
import pandas as pd


from agent import Assistant_Agent
from DatabaseManager import DatabaseManager
from models.models import RequestModel, ResponseModel
import json



uri = "./data/lancedb"
app = FastAPI()

db_manager = DatabaseManager(uri)
assistant = Assistant_Agent(db_manager)


@app.post("/assistant/")
async def get_response(request: RequestModel) -> ResponseModel:
    """
    Generate a response from the assistant agent.
    Returns a json with the answer and sources used.
    """

    response_stream = assistant.generate_response_stream(
        request = request

        )
    return StreamingResponse(response_stream, media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, log_level="info")

from typing import Union
from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import lancedb
import uvicorn
import pandas as pd


from components.agent import Assistant_Agent
from components.DatabaseManager import DatabaseManager
from models.models import RequestModel
import json

# setup
MODEL_PROVIDER = "open-ai"
MODEL_NAME = "gpt-4o"
DATABASE_LOCATION = "./data/lancedb"

app = FastAPI()
db_manager = DatabaseManager(DATABASE_LOCATION)

def get_assistant(request: RequestModel):
    language = request.metadata['language']
    return Assistant_Agent(db_manager, MODEL_PROVIDER, MODEL_NAME, language)

@app.post("/assistant/")
async def get_response(request: RequestModel, assistant: Assistant_Agent = Depends(get_assistant)):
    """
    Generate a response from the assistant agent.
    Returns a streaming response with the chatbot response and other metadata defined in the responsedict class.
    """

    response_stream = assistant.generate_response_stream(
        request = request

        )
    return StreamingResponse(response_stream, media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, log_level="info")

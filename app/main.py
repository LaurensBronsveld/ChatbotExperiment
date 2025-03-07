from typing import Union
from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse
import uvicorn

from components.agent import Assistant_Agent
from components.DatabaseManager import DatabaseManager
from models.models import RequestModel, StopModel
import logging

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

# Silence specific noisy libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logging.getLogger("pydantic_ai").setLevel(logging.WARNING)

DATABASE_LOCATION = "./data/lancedb"

app = FastAPI()
db_manager = DatabaseManager()

def get_assistant(request: RequestModel):
    language = request.metadata['language']
    return Assistant_Agent(db_manager, language)

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

@app.post("/assistant/stop/")
async def interrupt_response(stop_request: StopModel):
    logging.debug(stop_request)
    logging.debug(stop_request.session_id)
    if stop_request.session_id in Assistant_Agent.interrupts:
        Assistant_Agent.interrupts[stop_request.session_id].set()

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, log_level="info")

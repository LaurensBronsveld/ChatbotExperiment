from typing import Union
from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse
import uvicorn

from components.agent import Assistant
from components.DatabaseManager import DatabaseManager
from models.models import RequestModel

# setup

DATABASE_LOCATION = "./data/lancedb"

app = FastAPI()
db_manager = DatabaseManager()

def get_assistant(request: RequestModel):
    language = request.metadata['language']
    return Assistant(db_manager, language)

@app.post("/assistant/")
async def get_response(request: RequestModel, assistant: Assistant = Depends(get_assistant)):
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

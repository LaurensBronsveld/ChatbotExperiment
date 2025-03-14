from typing import Union
from fastapi import FastAPI, Request, Depends, APIRouter
from fastapi.responses import StreamingResponse
import uvicorn

from agents.agent import Assistant
from agents.BaseAgent import BaseAgent
from components.DatabaseManager import get_session
from models.models import RequestModel
import sys
import os

router = APIRouter()

def get_assistant(request: RequestModel):
    language = request.metadata['language']
    
    return BaseAgent(language)


@router.post("/assistant/")
async def get_response(request: RequestModel, agent: BaseAgent = Depends(get_assistant)):
    """
    Generate a response from the assistant agent.
    Returns a streaming response with the chatbot response and other metadata defined in the responsedict class.
    """
    
    response_stream = agent.generate_response_stream(
        request = request

        )
    return StreamingResponse(response_stream, media_type="text/event-stream")
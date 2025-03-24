from typing import Union
from fastapi import FastAPI, Request, Depends, APIRouter
from fastapi.responses import StreamingResponse
import uvicorn
from uuid import uuid4, UUID

from app.agents.agent import Assistant
from app.agents.BaseAgent import BaseAgent
from app.components.DatabaseManager import get_session
from app.api.chat.history import create_conversation
from app.models.models import *
from app.models.SQL_models import ConversationType, Conversation, ChatRole, ChatMessage
from sqlalchemy import select
import sys
import os
import logging
import json

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

# Silence specific noisy libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logging.getLogger("pydantic_ai").setLevel(logging.WARNING)



router = APIRouter()

def get_assistant(request: RequestModel):
    language = request.metadata['language']
    
    return BaseAgent(language)


@router.post("/chat/")
async def get_streaming_response(request: RequestModel, agent: BaseAgent = Depends(get_assistant)):
    """
    Creates a new conversation and generates a chat response from the agent.
    Posts the new conversation to the database
    Returns a streaming response with the chatbot response and other metadata defined in the responsedict class.
    """

    # create new conversation
    new_conversation = ConversationModel(
        session_id=uuid4(),
        user_id = request.user['user_id'],
        type = ConversationType.CHAT,
        title = 'title',
        metadata = request.metadata
    )
    await create_conversation(new_conversation)

    response_stream = agent.generate_response_stream(
        request = request,
        session_id = new_conversation.session_id

        )
    return StreamingResponse(response_stream, media_type="text/event-stream")

@router.post("/chat/{session_id}/")
async def get_follow_up_streaming_response(request: RequestModel, session_id: UUID, agent: BaseAgent = Depends(get_assistant)):
    """
    Generate a follow up response to an existing conversation.
    Returns a streaming response with the chatbot response and other metadata defined in the responsedict class.
    """
    
    response_stream = agent.generate_response_stream(
        request = request,
        session_id = session_id

        )
    return StreamingResponse(response_stream, media_type="text/event-stream")

@router.post("/chat_test/")
async def get_response(request: RequestModel, agent: BaseAgent = Depends(get_assistant)):
    """
    Creates a new conversation and generates a chat response from the agent.
    Posts the new conversation to the database
    Returns a streaming response with the chatbot response and other metadata defined in the responsedict class.
    """

    # create new conversation
    new_conversation = ConversationModel(
        session_id=uuid4(),
        user_id = request.user['user_id'],
        type = ConversationType.CHAT,
        title = 'title',
        metadata = request.metadata
    )
    await create_conversation(new_conversation)


    response = await agent.generate_response(
        request = request,
        session_id = new_conversation.session_id

        )
    return response

@router.post("/chat_test/{session_id}/")
async def get_follow_up_response(request: RequestModel, session_id: UUID, agent: BaseAgent = Depends(get_assistant)):
    """
    Generate a follow up response to an existing conversation.
    Returns a streaming response with the chatbot response and other metadata defined in the responsedict class.
    """
    
    response = await agent.generate_response(
        request = request,
        session_id = session_id

        )
    return response
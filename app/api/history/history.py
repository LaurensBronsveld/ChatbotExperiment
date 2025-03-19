from typing import Union
from fastapi import FastAPI, Request, Depends, APIRouter
from fastapi.responses import StreamingResponse
import uvicorn
from uuid import uuid4, UUID


from components.DatabaseManager import get_session
from models.models import *
from models.SQL_models import ConversationType, Conversation, ChatRole, ChatMessage
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

@router.post("/conversation/")
async def create_conversation(request: ConversationModel):
    try: 
        # get session
        db_generator = get_session()
        db = next(db_generator)

        # TODO: check if conversation with this session id already exists
        # should be impossible but maybe smart to double check regardless

        #create conversation SQL model
        new_conversation = Conversation(
            session_id = request.session_id,
            user_id = request.user_id,
            type =  request.type,
            title = request.title,
            chat_metadata = request.metadata.model_dump_json()
        )
        db.add(new_conversation)

    except Exception as e:
        logging.error(f"Something went wrong posting new conversation. error: {e} ")
    finally:
        db.commit()
        db.close()

@router.get("/conversation/")
async def get_all_conversations() -> ConversationModel:
    conversations = []
    try:
        #get session
        db_generator = get_session()
        db = next(db_generator)

        statement = select(Conversation)
        result = db.execute(statement)

        for conversation in result.scalars().all():
            conv_model = ConversationModel(session_id = conversation.session_id,
                                      user_id = conversation.user_id,
                                      type = conversation.type,
                                      title = conversation.title,
                                      metadata = json.loads(conversation.chat_metadata))
            conversations.append(conv_model)

        return conversations
    except Exception as e:
        logging.error(f"Error retrieving conversations: {e}")
    finally:
        db.close()

@router.get("/conversation/{session_id}/")
async def get_conversation(session_id: UUID):
    try:
        #get session
        db_generator = get_session()
        db = next(db_generator)

        statement = select(Conversation).where(Conversation.session_id == session_id)  
        result = db.execute(statement)
        conversation = result.scalars().one()
        conv_model = ConversationModel(session_id = conversation.session_id,
                                      user_id = conversation.user_id,
                                      type = conversation.type,
                                      title = conversation.title,
                                      metadata = json.loads(conversation.chat_metadata))

        return conv_model
    except Exception as e:
        logging.error(f"Error retrieving conversations: {e}")
    finally:
        db.close()

@router.get("/history/{session_id}/")
def get_history(session_id: UUID, show_system_calls: bool = False):
    try:
        db_generator = get_session()
        db = next(db_generator)
        conversation = db.query(Conversation).filter(Conversation.session_id == session_id).first()

        if not conversation:
            return f"session with id: {session_id} does not exist"

        dict = json.loads(conversation.chat_metadata)
        metadata = MetadataModel(**dict)

        result = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()

        messages = [] 
        for item in result:
            # skip system messages when fetching messages for front end
            if item.role == ChatRole.SYSTEM and show_system_calls == False:
                continue
            dict = json.loads(item.message)
            messages.append(MessageModel(**dict))
    
        history = ChatHistoryModel(
            metadata = metadata,
            messages = messages
        )
        
        return history
    except Exception as e:
        return f"somehing went wrong fetching history: {e}"
    finally:
        db.close()

from fastapi import APIRouter
from uuid import UUID


from app.components.DatabaseManager import get_session
from app.models.models import *
from app.models.SQL_models import Conversation, ChatRole, ChatMessage
from sqlalchemy import select
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
    """
    Creates a new conversation record in the database.

    Args:
        request (ConversationModel): The Pydantic model containing conversation details.
    """
    try:
        # get session
        db_generator = get_session()
        db = next(db_generator)

        # TODO: check if conversation with this session id already exists
        # should be impossible but maybe smart to double check regardless

        # create conversation SQL model
        new_conversation = Conversation(
            session_id=request.session_id,
            user_id=request.user_id,
            type=request.type,
            title=request.title,
            chat_metadata=request.metadata.model_dump_json(),
        )
        db.add(new_conversation)

    except Exception as e:
        logging.error(f"Something went wrong posting new conversation. error: {e} ")
    finally:
        db.commit()
        db.close()


@router.get("/conversation/")
async def get_all_conversations() -> list[ConversationModel]:
    """
    Retrieves all conversations from the database.

    Returns:
        list: A list of Pydantic ConversationModel objects.
              Returns an empty list if no conversations are found.

    """
    conversations = []
    try:
        # get session
        db_generator = get_session()
        db = next(db_generator)

        statement = select(Conversation)
        result = db.execute(statement)

        for conversation in result.scalars().all():
            conv_model = ConversationModel(
                session_id=conversation.session_id,
                user_id=conversation.user_id,
                type=conversation.type,
                title=conversation.title,
                metadata=json.loads(conversation.chat_metadata),
            )
            conversations.append(conv_model)

        return conversations
    except Exception as e:
        logging.error(f"Error retrieving conversations: {e}")
    finally:
        db.close()


@router.get("/conversation/{session_id}/")
async def get_conversation(session_id: UUID):
    """
    Retrieves a specific conversation by its session_id.

    Args:
        session_id (UUID): The unique identifier for the conversation.

    Returns:
        ConversationModel: The conversation details as a Pydantic model if found.
                           If the conversation is not found (e.g., due to .one()),
                           or an error occurs during processing or database interaction,
                           an exception will likely be raised and logged.
    """
    try:
        # get session
        db_generator = get_session()
        db = next(db_generator)

        # get all conversations
        statement = select(Conversation).where(Conversation.session_id == session_id)
        result = db.execute(statement)
        conversation = result.scalars().one()

        # remodel into Pydantic model
        conv_model = ConversationModel(
            session_id=conversation.session_id,
            user_id=conversation.user_id,
            type=conversation.type,
            title=conversation.title,
            metadata=json.loads(conversation.chat_metadata),
        )

        return conv_model
    except Exception as e:
        logging.error(f"Error retrieving conversations: {e}")
    finally:
        db.close()


@router.get("/history/{session_id}/")
def get_history(session_id: UUID, show_system_calls: bool = False):
    """
    Retrieves the chat history for a given session_id.

    Args:
        session_id (UUID): The unique identifier for the conversation session.
        show_system_calls (bool, optional): Whether to include system messages in the history.
                                           Defaults to False.

    Returns:
        ChatHistoryModel: The chat history including metadata and messages if found.
        str: An error message string if the session does not exist or an
             exception occurs during processing.
    """
    try:
        # get db session
        db_generator = get_session()
        db = next(db_generator)

        # get conversation with corresponding session_id
        conversation = (
            db.query(Conversation).filter(Conversation.session_id == session_id).first()
        )

        # exit if session does not exist
        if not conversation:
            return f"session with id: {session_id} does not exist"

        # get metadata from conversation
        metadata_dict = json.loads(conversation.chat_metadata)
        metadata = MetadataModel(**metadata_dict)

        # get all messages with the corresponding session_id
        result = (
            db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()
        )
        messages = []
        for item in result:
            # skip system messages when fetching messages for front end
            if item.role == ChatRole.SYSTEM and show_system_calls == False:
                continue
            message_dict = json.loads(item.message)
            messages.append(MessageModel(**message_dict))

        history = ChatHistoryModel(metadata=metadata, messages=messages)

        return history
    except Exception as e:
        return f"somehing went wrong fetching history: {e}"
    finally:
        db.close()

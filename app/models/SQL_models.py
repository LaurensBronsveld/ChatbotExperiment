from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
import uuid
import enum

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Text,
    VARCHAR,
    Enum
    
)

Base = declarative_base()

class ChatRole(enum.Enum): #Create an enum class
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ConversationType(enum.Enum):
    CHAT = "chat"
    IMAGE_GEN = "image generation"
    RESEARCH = "research"

# declare models
class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String, nullable=False)
    chunk = Column(String, nullable=False)
    embedding = Column(Vector(3072))  # Adjust vector dimension to match your embeddings
    
    class Config:
        orm_mode = True

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), index=True, nullable=False, unique=True, default=uuid.uuid4)
    user_id = Column(Integer)
    type = Column(Enum(ConversationType, name = "type of conversation"))
    title = Column(VARCHAR)
    chat_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    class Config:
        orm_mode = True

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("conversations.session_id"), index=True)
    role = Column(Enum(ChatRole, name="role_enum"))
    message = Column(JSON)
    language = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    class Config:
        orm_mode = True



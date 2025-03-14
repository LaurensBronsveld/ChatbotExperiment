from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
import uuid
from components.DatabaseManager import Base
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
    __tablename__ = "conversation"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    user_id = Column(Integer)
    title = Column(VARCHAR)
    chat_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    class Config:
        orm_mode = True

class ChatMessage(Base):
    __tablename__ = "chat_message"
    id = Column(Integer, primary_key=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("conversation.id"), index=True)
    role = Column(Enum)
    message = Column(JSON)
    created_at = Column(DateTime)
    language = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    class Config:
        orm_mode = True



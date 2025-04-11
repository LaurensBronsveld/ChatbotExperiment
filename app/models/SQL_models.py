from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.orm import declarative_base, mapper, relationship, Mapped, mapped_column
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
    Enum,
    Index,
    event
    
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

class DocumentType(enum.Enum):
    MD = '.md'
    TXT = '.txt'
    PDF = '.pdf'

class DocumentSubject(enum.Enum):
    FINANCE = "finance"
    HR = 'hr'
    TECH = 'tech'


# declare models
class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    chunk_metadata = Column(JSON)
    chunk = Column(String, nullable=False)
    embedding = Column(Vector(1536))  # vector for dense search, currently using embedding_small from OpenAI
    chunk_tsv = Column(TSVECTOR) # vector for sparse search
    document: Mapped["Document"] = relationship(back_populates="chunks") # creates a link to related document

    #indexes
    __table_args__ = (
        # Index for Full-Text Search on the tsvector column
        Index(
            'idx_gin_chunk_tsv',        # Index name
            'chunk_tsv',                # Column to index
            postgresql_using='gin'      # Index type (GIN is best for tsvector)
        ),
   
        # HNSW index
        Index(
            'idx_hnsw_embedding',       # Index name
            'embedding',                # Column to index
            postgresql_using='hnsw',    # Index type
            postgresql_with={'m': 16, 'ef_construction': 64}, # Example parameters (tune these)
            postgresql_ops={'embedding': 'vector_cosine_ops'} # Operator class (use cosine, l2, or ip based on your distance metric)
        ))
    
    def __init__(self, document_id, chunk, chunk_metadata=None, embedding=None):
        super().__init__(document_id=document_id, chunk=chunk, chunk_metadata=chunk_metadata, embedding=embedding)

    def __repr__(self):
        return f"<Chunk(id={self.id}, document_id={self.document_id}, chunk={self.chunk[:20]}...)>"

    class Config:
        orm_mode = True


@event.listens_for(Chunk, 'before_insert')
@event.listens_for(Chunk, 'before_update')
def update_tsv(mapper, connection, target):
    target.chunk_tsv = func.to_tsvector('english', target.chunk)

class Document(Base):
    __tablename__ = "documents"

    pk = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(UUID(as_uuid=True), nullable=False, unique=True)
    title = Column(String(255))
    type = Column(Enum(DocumentType), name = "filetype of document")
    subject = Column(Enum(DocumentSubject), name = "subject of document")
    location = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    children: Mapped[list["Chunk"]] = relationship()
    
    class Config:
        orm_mode = True

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), index=True, nullable=False, unique=True, default=uuid.uuid4)
    user_id = Column(Integer)
    type = Column(Enum(ConversationType, name = "type of conversation"))
    title = Column(String(255))
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
    language = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    class Config:
        orm_mode = True



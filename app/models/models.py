from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Required, NotRequired
from typing import Optional
from uuid import UUID, uuid4
from app.models.SQL_models import ConversationType, ChatRole

# class ResponseModel(BaseModel):
#     response: str = Field(description= "The answer generated by the LLM in response to the user's question")
#     sources: List[str] = Field(description= "List of sources used by the LLM to answer the question, should provide links to handbook pages it references")

# class QueryModel(BaseModel):
#     query: str = Field(description = "The question posed to the chatbot by the user")
#     use_rag: bool = Field(default = True, description = "Boolean whether the LLM should use RAG to answer the question")
#     use_ddrg: bool = Field(default = False, description = "Boolean whether the LLM should browse the web to answer the question")




# models for chat requests
class Tool(BaseModel):
    name: str 
    enabled: bool

class MetadataModel(BaseModel):
    language: str = Field(description = "Language in which the user's question is written and in which the Assistant should respond")
    session_id: Optional[UUID] = Field(description = "session id which can be used to retrieve conversation history later. Should be generated at the start of each new conversation")
    tools: list[Tool] = Field(description = "List of tools the LLM agent can use.")


class UserModel(BaseModel):
    user_id: int
    question: str   
    context: Optional[list[dict]] = []

class RequestModel(BaseModel):
    metadata: object
    user: object

# models for chat response
class SourceDict(TypedDict):
    id: Required[int]
    type: Required[str]
    url: NotRequired[Optional[str]]
    text: NotRequired[Optional[str]]
    uri: NotRequired[Optional[str]]
    used: Required[bool] 
# class SourceDict(TypedDict):
#     id: int = Field(description="ID of source used for citations and determining if the source is used or not")
#     type: str = Field(description="The type of source (e.g., url, snippet, file).")
#     url: Optional[str] = Field(None, description="The URL of the source if applicable.")
#     text: Optional[str] = Field(None, description="Text snippet from the source if applicable.")
#     uri: Optional[str] = Field(None, description="URI of a file if applicable.")
#     used: bool = Field(description="Indicates whether this source was used in the response.", default=False)

class ResponseModel(BaseModel):
    content: Optional[str] = Field(description="The answer generated by the LLM in response to the user's question.", default=None)  
    able_to_answer: Optional[bool] = Field(description="Indicates whether the LLM was able to generate a confident answer.", default=None)
    question_classification: Optional[str] = Field(description="The category or classification of the question.", default=None)
    follow_up_questions: Optional[list[str]] = Field(description="Suggested follow-up questions for the user.", default=None)

class ResponseMetadata(BaseModel):
    sources: list[SourceDict] = Field(description="List of sources referenced by the LLM to answer the question.")
    tools_used: list[str] = Field(description="List of tools that were used to generate the response.")
    session_id: UUID = Field(description="The session ID associated with the interaction.")
    trace_id: str = Field(description="The trace ID for debugging or monitoring purposes.")
  

# models for database requests
class SearchRequest(BaseModel):
    query: str
    tool_call_attempt: int
    limit: int = 5

# conversation history
class MessageModel(BaseModel):
    role: ChatRole
    content: str
    context: Optional[list[dict]] = []

class ChatHistoryModel(BaseModel):
    metadata: MetadataModel
    messages: list[MessageModel]

class ConversationModel(BaseModel):
    session_id: UUID
    user_id: int
    type: ConversationType
    title: str
    metadata: MetadataModel
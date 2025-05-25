from __future__ import annotations as _annotations

import httpx
import re
from typing import Any
from pydantic import ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ToolReturnPart
from app.agents.system_prompts import get_chatbot_prompt
from app.models.models import *
from app.models.SQL_models import *
from app.agents.LLMs import get_model
from app.components.DatabaseManager import get_session
from app.api.chat.history import get_history
from app.api.tools.search import search_database
from app.config import settings
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from uuid import uuid4
import logging


# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

# Silence specific noisy libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logging.getLogger("pydantic_ai").setLevel(logging.WARNING)


langfuse = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST,
)


class BaseAgent:
    def __init__(self, language: str = "en"):
        """
        Initializes the BaseAgent.

        Args:
            language (str, optional): The language for the agent's responses.
                                      Defaults to "en" (English).
        """
        self.language = language
        self.tools = []

    def get_chat_history(self, session_id: str):
        """
        Retrieves the chat history for a given session ID.

        Args:
            session_id (str): The unique identifier for the chat session.

        Returns:
            list[dict] | None: A list of message dictionaries if history is found,
                               each with "role" and "content". Returns None if
                               no history is found or an error occurs.
        """
        try:
            history = []
            results = get_history(session_id=session_id, show_system_calls=True)

            if results:
                for message in results.messages:
                    history.append(
                        {"role": message.role.value, "content": message.content}
                    )
                return history
            else:
                return None
        except Exception as e:
            logging.error(f"error retrieving chat history: {e}")
            return None 

    def update_chat_history(self, db: object, session_id: UUID, message: MessageModel):
        """
        Updates the chat history in the database with a new message.

        Args:
            db (object): The SQLAlchemy database session object.
            session_id (UUID): The unique identifier for the chat session.
            message (MessageModel): The Pydantic message model to add to the history.
        """
        try:
            # add new message to database

            new_chat_message = ChatMessage(
                session_id=session_id,
                role=message.role,
                message=message.model_dump_json(),
                language=self.language,
            )
            db.add(new_chat_message)
            db.commit()
        except Exception as e:
            logging.error(f"something went wrong updating history : {e}")

    def search_tool(
        self, ctx: RunContext, query: str, tool_call_attempt: int = 0, limit: int = 5
    ):
        """
        Searches the handbook database using the provided query for relevant chunks of information.

        This method performs a vector search on the database containing embedded chunks of the Gitlab Handbook.
        It returns the top X matching results from the database in the form of a JSON list of dictionaries with ID, source and content.

        Args:
            ctx (RunContext): The context of the current run, providing access to dependencies and state.
            query (str): The search query to use against the handbook database.
            tool_call_attempt (int, optional): The attempt number of the tool call, used to generate unique IDs for results.
                                               First attempt is 0. Defaults to 0.
            limit (int, optional): The maximum number of search results to return from the reranker.
                                   Defaults to 5.

        Returns:
            list[dict] or dict: A list of dictionaries, where each dictionary represents a search result
                            containing the 'id', 'source', and 'chunk'. Returns an empty list or
                            error dictionary if search fails.
        """
        # create search request
        request = SearchRequest(
            query=query, tool_call_attempt=tool_call_attempt, limit=limit
        )

        # search database
        result = search_database(request)

        return result

    def get_tool_results(
        self, ctx: RunContext, result: object, tool_name: str, db, session_id=None
    ):
        """
        Extracts and processes results from a tool call.

        Parses the tool return parts from the agent's result object, updates
        chat history with system messages about tool usage, and formats
        the tool content into a list of SourceDict objects.

        Args:
            ctx (RunContext): The context of the current run. (Niet direct gebruikt in de body)
            result (object): The result object from an agent run, expected to contain messages.
            tool_name (str): The name of the tool whose results are to be extracted.
            db (object): The SQLAlchemy database session object for updating history.
            session_id (UUID, optional): The session ID for updating chat history. Defaults to None.

        Returns:
            tuple[list[SourceDict], list[str]]: A tuple containing:
                - A list of SourceDict objects representing the formatted tool results.
                - A list of tool names that were called.
        """
        content = []
        sources = []
        tools = []

        # get resuls from tool call out of Result object
        for message in result.all_messages():
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == tool_name:
                    content.extend(part.content)
                    tools.append(part.tool_name)

                    if session_id:
                        system_message = MessageModel(
                            role=ChatRole.SYSTEM,
                            content=f"Called tool: {tool_name}. Results: {part.content}",
                        )

                        self.update_chat_history(db, session_id, system_message)

        # create source objects
        for source_item in content: 
            url = source_item["source"]
            id_val = source_item["id"]
            text = source_item["chunk"]


            url_regex = r"^(https?:\/\/|www\.)\S+$"  # regex which matches most urls starting with http(s)// or www.
            uri_regex = r"^(?:[a-zA-Z]:\\|\/)[^\s]*$"  # regex which matches absolute file paths in windows and unix systems

            if re.match(url_regex, url):
                sources.append(SourceDict(id=id_val, type="url", url=url, text=text))
            elif re.match(uri_regex, url):
                sources.append(SourceDict(id=id_val, type="file", uri=url, text=text))
            else:
                sources.append(SourceDict(id=id_val, type="snippet", text=text))

        return sources, tools

    @observe(
        capture_input=True,
        capture_output=True,
        as_type="generation",
        name="chatbot response",
    )
    async def process_answer(self, prompt: list, session_id: str, db):
        """
        Processes a user prompt, generates a response using an LLM agent with a search tool,
        and streams the response.

        This asynchronous generator initializes an agent, retrieves chat history,
        runs the agent with the history and prompt, processes tool calls (search),
        and yields metadata and structured response chunks. It also updates the
        chat history with the assistant's final response.

        Args:
            prompt (list): The user's input/prompt. 
            session_id (str): The unique identifier for the chat session.
            db (object): The SQLAlchemy database session object.

        Yields:
            str: JSON serialized strings of ResponseMetadata and ResponseModel chunks.
        """
        response = None
        old_content = ""

        trace_id = langfuse_context.get_current_trace_id()
        logging.debug(langfuse_context.get_current_trace_url())

        model = get_model()
        agent = Agent(
            model,
            result_type=ResponseModel,
            system_prompt=get_chatbot_prompt(self.language),
        )
        agent.tool(self.search_tool)
        history = self.get_chat_history(session_id)

        async with agent.run_stream(str(history)) as result:
            sources, tools_used = self.get_tool_results(
                self, 
                result=result,
                tool_name="search_tool",
                db=db,
                session_id=session_id,
            )

            metadata = ResponseMetadata(
                sources=sources,
                tools_used=tools_used,
                session_id=str(session_id),
                trace_id=trace_id,
            )
            yield (metadata.model_dump_json())

            async for structured_result, last in result.stream_structured(
                debounce_by=0.01
            ):
                try:
                    chunk = await result.validate_structured_result(
                        structured_result, allow_partial=not last
                    )

                    content = chunk.content
                    if not last:
                        if content != old_content:
                            old_content = content
                            # create response object
                            response = ResponseModel(
                                content=chunk.content,
                            )
                            yield (response.model_dump_json())
                    else:
                        # create response object
                        response = ResponseModel(
                            content=content,
                            able_to_answer=chunk.able_to_answer,
                            question_classification=chunk.question_classification,
                            follow_up_questions=chunk.follow_up_questions,
                        )
                        yield (response.model_dump_json())
                except ValidationError as exc:
                    if all(e["type"] == "missing" for e in exc.errors()):
                        logging.warning(
                            f"Missing field during streaming: {exc}"
                        )  # Log the warning.
                        continue
                    else:
                        raise
                except httpx.ReadError as e:
                    logging.error(f"Streaming interrupted: {e}")
                    break  # Stop streaming if connection is lost

            assistant_message = MessageModel(role=ChatRole.ASSISTANT, content=content)

            self.update_chat_history(db, session_id, assistant_message)

    async def generate_response_stream(
        self, request: RequestModel, session_id: UUID = None
    ):
        """
        Generates a streaming response for a chat request.

        Initializes a database session, sets up Langfuse tracing, updates chat history
        with the user's message, and then processes the answer using `process_answer`
        to yield response chunks.

        Args:
            request (RequestModel): The user's request containing the question and context.
            session_id (UUID, optional): The unique identifier for the chat session.
                                         Defaults to None (hoewel de logica het vereist).

        Yields:
            str: JSON serialized chunks of the response stream.
        """
        # get database session
        db_generator = get_session()
        db = next(db_generator)

        # start langfuse trace
        trace = langfuse.trace(name="chat_request")

        # add user question to history
        user_message = MessageModel(
            role=ChatRole.USER,
            content=request.user["question"],
            context=request.user["context"],
        )
        self.update_chat_history(db, session_id, user_message)

        # get streaming response from agent
        retries = 1
        old_chunk = ""
        for attempt in range(retries):
            try:
                async for chunk in self.process_answer(
                    request.user["question"], session_id, db
                ):
                    if chunk is not old_chunk:
                        old_chunk = chunk
                        yield chunk
                break
            except Exception as e:
                if attempt < retries - 1: 
                    logging.error(
                        f"Error: {e} occured while streaming response, repeating attempt"
                    )
                    continue
                else:
                    logging.error(
                        f"something went wrong generating the response: error: {e}"
                    )
                    db.rollback()  # rollback changes to the database

        # close database session
        db.close()

    async def generate_response(self, session_id, request: RequestModel):
        """
        Generates a non-streaming (complete) response for a chat request.

        Initializes a database session, updates chat history, runs the agent to get a full response,
        updates history with the assistant's response, processes tool results, and returns
        the metadata and response.

        Args:
            session_id (UUID): The unique identifier for the chat session.
                               (Type hint ontbreekt in method signature)
            request (RequestModel): The user's request containing the question and context.

        Returns:
            dict: A dictionary containing "metadata" and "response" JSON strings.
        """
        # get database session
        db_generator = get_session()
        db = next(db_generator)

        # start langfuse trace
        fake_trace = uuid4()
        # add user question to history
        user_message = MessageModel(
            role=ChatRole.USER,
            content=request.user["question"],
            context=request.user["context"],
        )
        self.update_chat_history(db, session_id, user_message)

        model = get_model()
        agent = Agent(
            model,
            result_type=ResponseModel,
            system_prompt=get_chatbot_prompt(self.language),
        )
        agent.tool(self.search_tool)

        history = self.get_chat_history(session_id)

        response = await agent.run(str(history))
        assistant_message = MessageModel(
            role=ChatRole.ASSISTANT, content=response.data.content
        )

        self.update_chat_history(db, session_id, assistant_message)

        sources, tools_used = self.get_tool_results(
            self, 
            result=response,
            tool_name="use_search_tool", 
            db=db,
            session_id=session_id,
        )

        metadata = ResponseMetadata(
            sources=sources,
            tools_used=tools_used,
            session_id=str(session_id),
            trace_id=str(fake_trace),
        )
        metadata_json = metadata.model_dump_json()

        # commit and close database session
        db.close()
        response_json = response.data.model_dump_json()
        return {"metadata": metadata_json, "response": response_json}

    def generate_simple_response(self, request: str) -> dict[str, Any]:
        """
        Generates a simple, non-streaming response for a given request string.

        Initializes a database session, sets up an agent with a search tool,
        runs the agent synchronously, processes tool results, and returns
        the answer and sources.

     

        Args:
            request (str): The user's request string.

        Returns:
            dict: A dictionary containing "answer" (ResponseModel data) and "sources" (list of SourceDict).
        """
        # Get database session
        db_generator = get_session()
        db = next(db_generator)

        # Get model and setup agent
        model = get_model()
        agent = Agent(model, system_prompt=get_chatbot_prompt(self.language))
        agent.tool(self.search_tool)

        response = agent.run_sync(request)
        sources, tools_used = self.get_tool_results( 
            self, 
            result=response,
            tool_name="search_tool",
            db=db
        )

        # Close database session
        db.close()

        # Just return the content string
        return {"answer": response.data, "sources": sources}
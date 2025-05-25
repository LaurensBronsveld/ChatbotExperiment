import tiktoken
import os
import fitz  # pymupdf
import pymupdf4llm # Niet gebruikt in deze file, mogelijk bedoeld voor pdf_document.markdown()

from openai import OpenAI # Niet direct gebruikt, maar OpenAI client wordt later geïnstantieerd.
from uuid import uuid4

from app.config import settings
from app.models.SQL_models import Document, DocumentType, Chunk
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextChunker:
    docs = []
    chunks = []

    def __init__(
        self,
        data_path,
        get_session_func,
        min_tokens=200,
        max_tokens=1000,
        overlap_tokens=100,
    ):
        """
        Initializes the TextChunker.

        Args:
            data_path (str): The path to the directory containing data to be processed.
            get_session_func (callable): A function that returns a database session when called.
            min_tokens (int, optional): The minimum number of tokens a chunk should ideally have.
                                        Defaults to 200. 
            max_tokens (int, optional): The maximum number of tokens for a chunk before splitting.
                                        Defaults to 1000.
            overlap_tokens (int, optional): The number of tokens to overlap between consecutive chunks.
                                            Defaults to 100.
        """
        self.get_session = get_session_func
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.encoding_for_model(settings.EMBED_MODEL)
        self.data_path = data_path
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens, chunk_overlap=overlap_tokens
        ).from_tiktoken_encoder(model_name=settings.EMBED_MODEL)

    def __enter__(self):
        """
        Context manager entry point. Obtains a database session and runs the data pipeline.

        Returns:
            TextChunker: The instance of the TextChunker itself.
        """
        self.db = next(self.get_session())
        self.run_data_pipeline()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit point. Commits or rolls back the database session and closes it.

        Args:
            exc_type (type | None): The type of the exception if one occurred, otherwise None.
            exc_value (Exception | None): The exception instance if one occurred, otherwise None.
            traceback (traceback | None): A traceback object if an exception occurred, otherwise None.
        """
        if exc_type:
            self.db.rollback()
        else:
            self.db.commit()
        self.db.close()

    def add_chunk(self, document_id, chunk, metadata=""):
        """
        Adds a text chunk to the internal list of chunks (self.chunks).

        If the provided chunk (with metadata prepended) exceeds `self.max_tokens`,
        it is further split using `self.splitter`. Otherwise, it's added as a single chunk.

        Args:
            document_id (UUID): The ID of the document this chunk belongs to.
            chunk (str): The text content of the chunk.
            metadata (str, optional): Metadata string to be prepended to the chunk.
                                      Defaults to "".
        """

        processed_chunk_text = f"{metadata} \n {chunk}"
        if len(self.tokenizer.encode(processed_chunk_text)) < self.max_tokens:
            self.chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_metadata=metadata,
                    chunk=processed_chunk_text.strip(), 
                )
            )
        else:
            split_chunks = self.splitter.split_text(processed_chunk_text) 
            for part in split_chunks:
                self.chunks.append(
                    Chunk(
                        document_id=document_id,
                        chunk_metadata=metadata, 
                        chunk=part.strip(),
                    )
                )

    def extract_text(self):
        """
        Extracts text from files in the specified `self.data_path`.

        Walks through the directory, processes files with allowed extensions (.md, .txt, .pdf),
        creates Document records in the database, and appends document content to `self.docs`.
        PDF conversion uses PyMuPDF4LLM.
        Commits Document records to the database.
        """
        allowed_extensions = {".md", ".txt", ".pdf"}  # Add more extensions as needed

        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                print(file)
                file_path = os.path.join(root, file)
                title, file_extension = os.path.splitext(file)  # Get the file extension
                if (
                    file_extension.lower() in allowed_extensions
                ):  # Check if extension is allowed
                    try:
                        doc_id_val = uuid4() 
                        doc = Document(
                            id=doc_id_val, 
                            title=title,
                            type=DocumentType(file_extension.lower()),
                            location=file_path,
                        )
                        self.db.add(doc)

                        if file_extension.lower() == ".pdf":
                            # Convert PDF to Markdown using PyMuPDF4LLM
                            content = pymupdf4llm.to_markdown(file_path)
                        else:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                        self.docs.append({"doc_id": doc.id, "text": content})

                    except Exception as e:
                        print(f"Warning: something went wrong with {file_path}: {e}")
                else:
                    print(f"Skipping file {file} due to unsupported extension.")

        # commit documents so that we can link the chunks later
        self.db.commit()

    def split_text_into_chunks(self):
        """
        Splits the text content of each document in `self.docs` into chunks.

        Iterates through `self.docs` and calls `self.add_chunk` for the text
        content of each document. The `add_chunk` method handles the actual
        splitting and storage of chunks in `self.chunks`.
        """
        for doc in self.docs:
            doc_id = doc["doc_id"]
            text = doc["text"]

            self.add_chunk(doc_id, text) 

    def create_embeddings(self, batch_size=500):
        """
        Creates vector embeddings for all chunks in `self.chunks` using OpenAI's API.

        Processes chunks in batches of the specified `batch_size`.
        The generated embeddings are assigned to the `embedding` attribute of each Chunk object.

        Args:
            batch_size (int, optional): The number of chunks to process in each API call.
                                        Defaults to 500.
        """
        client = OpenAI()
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i : i + batch_size]
            texts = [chunk.chunk for chunk in batch]
            try:
                response = client.embeddings.create(
                    input=texts, model=settings.EMBED_MODEL
                )
                print("batch done")
                # Extract the embeddings from the response
                embeddings = [entry.embedding for entry in response.data]
                for chunk, embedding in zip(batch, embeddings):
                    chunk.embedding = embedding
            except Exception as e:
                print(f"Embedding failed with error: {e}")

    def add_chunks_to_db(self):
        """
        Adds all Chunk objects currently in `self.chunks` to the database session.
        """
        for chunk in self.chunks:
            self.db.add(chunk)

    def clear_db(self):
        """
        Deletes all entries from the Chunk and Document tables in the database.

        """
        self.db.query(Chunk).delete()
        self.db.query(Document).delete()
        self.db.commit()

    def run_data_pipeline(self):
        """
        Executes the complete data processing pipeline:
        1. Extracts text from source files.
        2. Splits extracted text into chunks.
        3. Creates embeddings for these chunks.
        4. Adds the processed chunks to the database session.
        """
        # print("run_data_pipeline is called")
        self.extract_text()
        self.split_text_into_chunks()
        self.create_embeddings()
        self.add_chunks_to_db()
import tiktoken
import os
import fitz  # pymupdf
import pymupdf4llm

from openai import OpenAI
from uuid import UUID, uuid4
from pymupdf4llm import to_markdown

from app.config import settings
from app.models.SQL_models import Document, DocumentSubject, DocumentType, Chunk
from app.components.DatabaseManager import get_session
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextChunker:
    docs = []
    chunks = []

    def __init__(self, data_path, get_session_func, min_tokens=200, max_tokens=1000, overlap_tokens=100):
            self.get_session = get_session_func
            self.min_tokens = min_tokens
            self.max_tokens = max_tokens
            self.overlap_tokens = overlap_tokens
            self.tokenizer = tiktoken.encoding_for_model(settings.EMBED_MODEL)
            self.data_path = data_path
            self.splitter = RecursiveCharacterTextSplitter(chunk_size = max_tokens, chunk_overlap = overlap_tokens).from_tiktoken_encoder(model_name=settings.EMBED_MODEL)

    
    def __enter__(self):
        self.db = next(self.get_session())
        self.run_data_pipeline()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            self.db.rollback()
        else:
            self.db.commit()
        self.db.close()

    def add_chunk(self, document_id, chunk, metadata = ""):
        chunk = f"{metadata} \n {chunk}"
        if len(self.tokenizer.encode(chunk)) < self.max_tokens:
                self.chunks.append(Chunk(
                    document_id=document_id,
                    chunk_metadata=metadata,
                    chunk=chunk.strip()
                ))
        else:
            split_chunks = self.splitter.split_text(chunk)
            for part in split_chunks:
                self.chunks.append(Chunk(
                document_id=document_id,
                chunk_metadata=metadata,
                chunk=part.strip()
            ))
                

    def extract_text(self):
        """Extracts text from specified file types."""
        allowed_extensions = {".md", ".txt", ".pdf"}  # Add more extensions as needed

        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                file_path = os.path.join(root, file)
                title, file_extension = os.path.splitext(file)  # Get the file extension
                if file_extension.lower() in allowed_extensions:  # Check if extension is allowed
                    try:

                        doc = Document(
                            id=uuid4(),
                            title=title,
                            type = DocumentType(file_extension),
                            location=file_path
                        )
                        self.db.add(doc)
                        
                        # pdf conversion is still untested.
                        if file_extension.lower() == ".pdf":
                            # Convert PDF to Markdown using PyMuPDF4LLM
                            pdf_document = fitz.open(file_path)
                            content = pymupdf4llm.to_markdown(file_path)
                        else:
                            with open(file_path, 'r', encoding="utf-8") as f:
                                content = f.read()
                        self.docs.append({"doc_id": doc.id, "text": content})

                    except Exception as e:
                        print(f"Warning: something went wrong with {file_path}: {e}")
                else:
                    print(f"Skipping file {file} due to unsupported extension.")

        # commit documents so that we can link the chunks later
        self.db.commit()

    def split_text_into_chunks(self):
        for doc in self.docs:
            doc_id = doc['doc_id']
            text = doc['text']

            self.add_chunk(doc_id, text)

    def create_embeddings(self, batch_size = 500):
        client = OpenAI()
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i+batch_size]
            texts = [chunk.chunk for chunk in batch]
            try:
                response = client.embeddings.create(input = texts, model=settings.EMBED_MODEL)
                print("batch done")
                            # Extract the embeddings from the response
                embeddings = [entry.embedding for entry in response.data]
                for chunk, embedding in zip(batch, embeddings):
                    chunk.embedding = embedding
            except Exception as e:
                print(f"Embedding failed with error: {e}")
    
    def add_chunks_to_db(self):
        for chunk in self.chunks:
            self.db.add(chunk)

    def clear_db(self):
        self.db.query(Chunk).delete()
        self.db.query(Document).delete()
        self.db.commit()

    def run_data_pipeline(self):
        # print("run_data_pipeline is called")
        self.extract_text()
        self.split_text_into_chunks()
        self.create_embeddings()
        self.add_chunks_to_db()

    

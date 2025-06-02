from pydantic_settings import BaseSettings
from pydantic import ConfigDict
import os


class Settings(BaseSettings):
    PROJECT_NAME: str = "chat_demo"
    PROJECT_VERSION: str = "0.1.0"
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str
    ANTHROPIC_API_KEY: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    COHERE_API_KEY: str

    DATABASE_LOCATION: str = (
        "postgresql+psycopg://postgres:password@localhost:5432/handbook_db"
    )
    TEST_DATABASE_URL: str = (
        "postgresql+psycopg://postgres:password@localhost:5432/test_db"
    )
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    API_URL: str = "http://127.0.0.1:8000"
    LLM_PROVIDER: str = "open-AI"
    LLM_MODEL: str = "gpt-4o"
    EMBED_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSION: int = 2000
    COHERE_RERANK_MODEL: str = "rerank-v3.5"
    DATA_PATH: str = "data/handbook-main-content"

    model_config = ConfigDict(
        env_file=".env",
        extra="allow",
        arbitrary_types_allowed=True,
    )


settings = Settings()

# Set environment variables for LangFuse. This is needed for some reason, even if you pass the keys as parameters at intialisation
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY

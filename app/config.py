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
    # DATABASE_URL: str           # not used yet
    DATABASE_LOCATION: str = "./data/lancedb"
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    HANDBOOK_TABLE: str = "embedded_handbook_with_urls"
    HISTORY_TABLE: str = "history_table"
    LLM_PROVIDER: str = "open-AI"
    LLM_MODEL: str = "gpt-4o"
    EMBED_MODEL: str = "text-embedding-3-large"

    model_config = ConfigDict(
        env_file=".env",
        extra='allow',
        arbitrary_types_allowed=True,
    )

settings = Settings()

# Set environment variables for LangFuse. This is needed for some reason, even if you pass the keys as parameters at intialisation
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
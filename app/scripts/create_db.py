import sys
import asyncio
from app.core.DatabaseManager import get_session, recreate_tables
from app.components.HandbookTextChunker import HandbookTextChunker
from app.core.config import settings


async def create_database(mode: str = "append"):
    """
    Initializes or updates the database with data from the specified data_path.

    This function can operate in two modes:
    - "create": Drops all existing tables and recreates them before processing data.
    - "append": Adds new data to the existing database tables.

    It uses the HandbookTextChunker to process files from the data folder,
    which handles text extraction, chunking, embedding, and database insertion
    within its context manager.

    Args:
        mode (str, optional): The mode of operation. Can be "create" or "append".
                              Defaults to "append". If an invalid mode is provided,
                              the script will exit.
    """
    data_path = settings.DATA_PATH

    if mode.lower() == "create":
        recreate_tables()
        print("Old database tables dropped and recreated.")
    elif mode.lower() == "append":
        print("Adding new entries to existing database.")
    else:
        print("Invalid argument. Please use 'append' or 'create'.")
        sys.exit(1)

    # go through data pipeline
    async with HandbookTextChunker(
        data_path=data_path, get_session_func=get_session
    ) as chunker:
        pass


if __name__ == "__main__":
    mode = "append"  # Default: append

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    asyncio.run(create_database(mode=mode))

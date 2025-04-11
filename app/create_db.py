import sys
from app.config import settings
from app.components.DatabaseManager import get_session, recreate_tables
from app.components.TextChunker import TextChunker
from app.components.HandbookTextChunker import HandbookTextChunker

def create_database(mode: str = "append"):
    data_path = "data/handbook-main-content"

    if mode.lower() == "create":
        recreate_tables()
        print("Old database tables dropped and recreated.")
    elif mode.lower() == "append":
        print("Adding new entries to existing database.")
    else:
        print("Invalid argument. Please use 'append' or 'create'.")
        sys.exit(1)

    # go through data pipeline
    with HandbookTextChunker(data_path= data_path, get_session_func=get_session) as chunker:
        pass


if __name__ == "__main__":
    print(TextChunker)
    mode = "append"  # Default: append

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    create_database(mode=mode)

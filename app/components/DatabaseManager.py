from app.models.models import *
import logging
from app.config import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from app.models.SQL_models import Base


# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

# Silence specific noisy libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logging.getLogger("pydantic_ai").setLevel(logging.WARNING)


uri = settings.DATABASE_LOCATION
engine = create_engine(
    url=settings.DATABASE_LOCATION,
)
sessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session_factory = scoped_session(sessionLocal)
Base.metadata.create_all(engine)



def get_session():
    """
    Provides a SQLAlchemy database session.

    This function is a generator that yields a new database session
    from the scoped session factory. It ensures that the session is
    closed after its use.

    Yields:
        scoped_session: A SQLAlchemy database session object.
    """
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def recreate_tables():
    """
    Drops all tables defined in Base.metadata and recreates them.

    """
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
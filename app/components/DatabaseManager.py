

import pandas as pd
from app.models.models import *
import logging
from app.config import settings
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker, Session, scoped_session
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
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
    url = settings.DATABASE_LOCATION,
    )
sessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session_factory = scoped_session(sessionLocal)
Base.metadata.create_all(engine)
       
def get_session():
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def recreate_tables():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

# def get_table(table_name: str):
#         return db.open_table(table_name)
    
# def drop_table(self, table_name:str):
#         db.drop_table('table_name')

# def copy_table(table_name:str, columns_to_drop: list[str], remove_columns: bool = False):
#     original_table = db.open_table(table_name)
#     df = original_table.to_pandas()
#     if remove_columns:
#         df = df.drop(columns = columns_to_drop)
#     return db.create_table('copied_table', data=df)
        
# def get_tables(self):
#         return self.db.table_names()
    
# def update_table(self, table_name: str, session_id: str, share_token: str, new_history: str):
#         table = self.db.open_table(table_name)
#         data = [ChatHistory(session_id=session_id, share_token=share_token, history=new_history)]
#         table.add(data)

        
#     def update_chat_history(self, table_name: str, session_id: str, share_token: str, new_history: str):
#         table = self.db.open_table(table_name)
#         results = table.search().where(f"session_id = '{session_id}'").limit(1).to_list()
#         if results:
#             table.update(where=f"session_id = '{session_id}'", values={'history': new_history})
#             logging.debug("testdb update")
#         else: 
#             table.add([{"session_id" : session_id, "share_token": share_token, "history": new_history}])
#             logging.debug("testdb create")
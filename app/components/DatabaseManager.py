from typing import Optional
import lancedb
import pandas as pd
from models.models import *
import logging



# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

# Silence specific noisy libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logging.getLogger("pydantic_ai").setLevel(logging.WARNING)



class DatabaseManager:
    _instance: Optional['DatabaseManager'] = None #makes class singleton

    def __init__(self, uri: str):
        self.uri = uri
        self.db = lancedb.connect(uri)
    
    #creates instance of databasemanager or returns if one already exists.
    @classmethod
    def get_instance(cls, uri: str) -> "DatabaseManager":
        if cls._instance == None:
            cls._instance = cls(uri)
        return cls._instance
    
    def get_table(self, table_name: str):
        return self.db.open_table(table_name)
    
    def drop_table(self, table_name:str):
        self.db.drop_table('table_name')

    def copy_table(self, table_name:str, columns_to_drop: list[str], remove_columns: bool = False):
        original_table = self.db.open_table(table_name)
        df = original_table.to_pandas()
        if remove_columns:
            df = df.drop(columns = columns_to_drop)
        return self.db.create_table('copied_table', data=df)
        
    def get_tables(self):
        return self.db.table_names()
    
    def update_table(self, table_name: str, session_id: str, share_token: str, new_history: str):
        table = self.db.open_table(table_name)
        data = [ChatHistory(session_id=session_id, share_token=share_token, history=new_history)]
        table.add(data)

        
    def update_chat_history(self, table_name: str, session_id: str, share_token: str, new_history: str):
        table = self.db.open_table(table_name)
        results = table.search().where(f"session_id = '{session_id}'").limit(1).to_list()
        if results:
            table.update(where=f"session_id = '{session_id}'", values={'history': new_history})
            logging.debug("testdb update")
        else: 
            table.add([{"session_id" : session_id, "share_token": share_token, "history": new_history}])
            logging.debug("testdb create")
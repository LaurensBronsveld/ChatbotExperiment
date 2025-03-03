from typing import Optional
import lancedb
import pandas as pd

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
        

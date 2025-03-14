from __future__ import annotations as _annotations

from fastapi import APIRouter
from models.SQL_models import *
from components.DatabaseManager import get_session
from config import settings
import cohere
import logging
from sqlalchemy.orm import declarative_base, sessionmaker, Session, scoped_session
import openai
import numpy as np

router = APIRouter()



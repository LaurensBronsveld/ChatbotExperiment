from fastapi import APIRouter

from app.api.chat.chatV1 import router as chat_router
from app.api.tools.tools import router as tool_router
from app.api.chat.history import router as history_router
API_STR = "/api"

api_router = APIRouter(prefix=API_STR)
api_router.include_router(chat_router)
api_router.include_router(tool_router)
api_router.include_router(history_router)
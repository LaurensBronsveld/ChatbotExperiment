from fastapi import APIRouter

from api.chat.assistant import router as chat_router
from api.tools.tools import router as tool_router
API_STR = "/api"

api_router = APIRouter(prefix=API_STR)
api_router.include_router(chat_router)
api_router.include_router(tool_router)

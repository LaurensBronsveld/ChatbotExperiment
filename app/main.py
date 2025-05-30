from fastapi import FastAPI, APIRouter
import uvicorn

from app.api import api_router


main_router = APIRouter()

# setup
app = FastAPI(title="GitLab Handbook Chatbot")

app.include_router(main_router)
app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, log_level="info")

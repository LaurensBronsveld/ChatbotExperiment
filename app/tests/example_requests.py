from uuid import UUID


def get_request(prompt: str, session_id: UUID = None):
    request = {
        "metadata": {
            "language": "nl",
            "session_id": session_id,
            "tools": [{"name": "RAG", "enabled": True}],
        },
        "user": {"user_id": 1, "question": prompt, "context": []},
    }
    return request

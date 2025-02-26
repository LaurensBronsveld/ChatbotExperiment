from fastapi.testclient import TestClient

import sys
import os

# Add the root directory to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../app'))

from app.main import app

# FILE: app/test_main.py

client = TestClient(app)

def test_streaming_response():
    response = client.post("/assistant/", json={"query": "What is GitLab's approach to paid time off (PTO)"})
    
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/json'
    
    for chunk in response.iter_content(chunk_size=8192):
        print(chunk.decode('utf-8'))

if __name__ == "__main__":
    test_streaming_response()
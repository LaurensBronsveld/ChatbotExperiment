import requests

API_URL = "http://127.0.0.1:8000/assistant/"

def test_streaming_response():
    response = requests.post(API_URL, data = "What is GitLab's approach to paid time off (PTO)",stream=True
)

    for chunk in response.iter_content(chunk_size=8192):
        print(chunk.decode('utf-8'))

if __name__ == "__main__":
    test_streaming_response()
# Handbook chatbot

## backend

to run the API server:

1. create virtual environment: uv venv
2. activate environment: .venv\Scripts\activate
3. install dependencies: uv pip install -r app/pyproject.toml
4. (optional) create .env file and set API keys if not set globally
5. start API server: uvicorn app.main:app 

## frontend
If you want to use streamlit to test the API server:

1. install dependencies: uv pip install -r frontend/pyproject.toml
2. start Streamlit: streamlit run frontend/main.py


To build the RAG database run this command:
    python -m app/scripts/create_db.py     
requires gitlab handbook files from: https://gitlab.com/gitlab-com/content-sites/handbook/-/tree/main/content stored in data folder in root directory.

To run the evaluation pipeline run this command:
    python -m app/scripts/evaluate_answer
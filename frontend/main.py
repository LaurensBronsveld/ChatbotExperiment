import streamlit as st
import requests
import asyncio
import time
import json

API_URL = "http://127.0.0.1:8000"

def get_response(query: str, use_rag = True, use_ddrg = False):
    try: 
        sources = []
        result = requests.post(
            f"{API_URL}/assistant/",
            json={
                "query": query,
                'use_rag': use_rag,
                "use_ddrg": use_ddrg
            }
            )
       
        data = result.json()
        response = data['response']
        sources = data['sources']
       
        return response
    except Exception as e:
        print({"error": e})
        
def get_streaming_resonse(query: str, use_rag = True, use_ddrg = False):
    try:
        response = requests.post(
            f"{API_URL}/assistant/",
            json={
                "query": query,
                'use_rag': use_rag,
                "use_ddrg": use_ddrg
            },
            stream=True,
            headers = {'Accept': 'text/event-stream'}
            )
        print(response)
        return response
    except Exception as e:
        print({"error": e})

def main():
    st.title("chatbot experiment")
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Handle new messages
    if prompt := st.chat_input("ask a question"):

        # Display user message
        with st.chat_message('user'):
            st.markdown(prompt)

        st.session_state.messages.append({"role": 'user', 'content': prompt})

        # Display assistant response
        try:
            response_stream = get_streaming_resonse(prompt)
            sources = []
            if response_stream:
                with st.chat_message('assistant'):
                    message_placeholder = st.empty()
                    full_response = ""
                    data = ""
                    #process streaming response
                    for chunk in response_stream.iter_content(chunk_size=None, decode_unicode=True):
                        data = json.loads(chunk)
                        if data['response']:
                                response = data['response']
                
                                full_response+=response #add chunk to response
                                message_placeholder.markdown(full_response) #display updated response
                   
                    sources = data['sources']       

                    st.session_state.messages.append({"role": 'assistant', 'content': full_response})
                
                with st.sidebar:
                    for source in sources:
                        st.write(source)
        except Exception as e:
            print("something went wrong with the response. Error: " + e)

        
   
if __name__ == "__main__":
    main()
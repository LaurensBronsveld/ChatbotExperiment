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
       
        return response, sources
    except Exception as e:
        print({"error": e})
        

# stream response to simulate typing. This is kinda slow so now so it is not used.
def stream_response(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

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
            response, sources = get_response(prompt)
            with st.chat_message('assistant'):
                st.markdown(response)
            
            for source in sources:
                with st.sidebar:
                    
                    st.write(source)

            st.session_state.messages.append({"role": 'assistant', 'content': response})
        except Exception as e:
            print("something went wrong with the response. Error: " + e)

        
   
if __name__ == "__main__":
    main()
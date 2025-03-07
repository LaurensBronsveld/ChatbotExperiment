import streamlit as st
import requests
import json


API_URL = "http://127.0.0.1:8000"


def get_response(query: str):
    try: 
        sources = []
        result = requests.post(
            f"{API_URL}/assistant/",
            json={"metadata": {"language": "nl",
                        "session_id": None,
                        "tools": [{ "name": "HR", "enabled": True }]
                    }, 
                    "user": {
                        "question": "Hoeveel dagen vakantie heb ik per jaar?",
                        "context": [
                            { "type": "file", "URL": "" }, 
                            { "type": "snippet", "text": ""},
                            { "type": "url", "url": "https://example.com" }
                        ]
                    }
                    }
            )
       
        data = result.json()
        response = data['response']
        sources = data['sources']
       
        return response
    except Exception as e:
        print({"error": e})
        
def get_streaming_resonse(query: str, session_id: str):
    try:
        response = requests.post(
            f"{API_URL}/assistant/",
            json={"metadata": {"language": "nl",
                        "session_id": session_id,
                        "tools": [{ "name": "HR", "enabled": True }]
                    }, 
                    "user": {
                        "question": query,
                        "context": [
                            { "type": "file", "URL": "" }, 
                            { "type": "snippet", "text": ""},
                            { "type": "url", "url": "https://example.com" }
                        ]
                    }
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

    # Initialize session_id in session_state if it doesn't exist
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

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
            response_stream = get_streaming_resonse(prompt, st.session_state.session_id)
            sources = []
            metadata_received = False

            if response_stream:
                with st.chat_message('assistant'):
                    message_placeholder = st.empty()
                    full_response = ""
                    data = ""
                    last_displayed = ""

                    #process streaming response
                    for chunk in response_stream.iter_content(chunk_size=None, decode_unicode=True):
                        data = json.loads(chunk)
                        print(data)
                        # first chunk is metadata
                        if not metadata_received:
                            sources = data['sources']
                            st.session_state.session_id = data['session_id']
                            metadata_received = True  
                            continue       
                        
                        if data['content']:
                            response = data['content']
                
                             # Extract only new content
                            new_data = response[len(last_displayed):]  
                            last_displayed = response  # Update tracking

                            if new_data:  # Only update UI if there's actually new text
                                full_response += new_data
                                message_placeholder.markdown(full_response)
                   
                    
                    questions = data['follow_up_questions']
                       
                    st.session_state.messages.append({"role": 'assistant', 'content': full_response})
                
                with st.sidebar:
                    for source in sources:
                        st.write(source)
                
                with st.container():
                    for question in questions:
                        st.write(question)

                
        except Exception as e:
            print(f"something went wrong with the response. Error: {e}")

        
   
if __name__ == "__main__":
    main()
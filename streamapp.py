import streamlit as st
from huggingface_hub import InferenceClient
import toml

# Load the TOML configuration file
config = toml.load("config.toml")

# Streamed response emulator
def response_generator(messages):
    # Get the Hugging Face API key from the TOML file
    token = config["secrets"]["hugging_face_api_key"]
    
    # Initialize the InferenceClient with the token
    client = InferenceClient(
        "microsoft/Phi-3-mini-4k-instruct",
        token=token,
    )
    
    # Generating response using the entire conversation history
    response = "".join(
        message.choices[0].delta.content
        for message in client.chat_completion(
            messages=messages,
            max_tokens=500,
            stream=True
        )
    )
    return response

st.title("Simple Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display the assistant's response using the entire chat history
    response = response_generator(st.session_state.messages)
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

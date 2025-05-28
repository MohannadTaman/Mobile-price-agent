import streamlit as st
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from main_chatbot import create_chatbot_with_memory_and_tools  # Import the chatbot creation function

# Initialize the chatbot
chatbot = create_chatbot_with_memory_and_tools()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What's up?"):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    try:
        response = chatbot.invoke({"input": prompt})["output"]
    except Exception as e:
        response = f"An error occurred: {e}"

    # Display assistant response in chat message container
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
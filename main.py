import streamlit as st
import json
import logging
import tiktoken
import time
from datetime import datetime
from rag_chatbot import handle_user_input
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from streamlit_float import *

# Set up logging
logger = logging.getLogger()
logging.basicConfig(encoding="UTF-8", level=logging.INFO)

# Define avatar URLs
USER_AVATAR = "👤"
ASSISTANT_AVATAR = "https://raw.githubusercontent.com/j0ntendo/med-rag-chatbot/main/logo_transparent.png"

logo = "https://github.com/j0ntendo/med-rag-chatbot/blob/main/chatbot_logo.jpg?raw=true"
url1 = "https://simonezz.tistory.com/41"
url2 = "https://wikidocs.net/231585"

# Title and Sidebar
st.title("DentaReAct Chatbot 🦷")
st.write("Welcome to DentaReAct! I use a ReAct agent to answer your dental questions based on expert-run forums. If needed, I'll search the internet for additional information.")
st.sidebar.image(logo, width=150, use_column_width=True)
st.sidebar.write("🛠️I have access to the following tools🛠️:")
st.sidebar.markdown(f"- [Okapi BM25 Retriever]({url1})")
st.sidebar.markdown(f"- [Maximal Marginal Relevance]({url2})")
st.sidebar.markdown("- DuckDuckGo Retriever")

# Initialize session state
if 'generated_history' not in st.session_state:
    st.session_state['generated_history'] = []
if 'user_history' not in st.session_state:
    st.session_state['user_history'] = []
if 'stored_sessions' not in st.session_state:
    st.session_state['stored_sessions'] = []
if 'input' not in st.session_state:
    st.session_state['input'] = ''
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=10, return_messages=True)
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Feedback logging function
def log_feedback(icon):
    st.toast("Thanks for your feedback!", icon="👌")
    if len(st.session_state["messages"]) >= 2:
        last_messages = json.dumps(st.session_state["messages"][-2:])
    else:
        last_messages = "Not enough messages to log."

    activity = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": "
    activity += "positive" if icon == "👍" else "negative"
    activity += ": " + last_messages

    logger.info(activity)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask me anything dental related"):
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response from chatbot
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st_callback = StreamlitCallbackHandler(st.container())
        response = handle_user_input(prompt)
        output_text = response.get('output', '')

        st.markdown(output_text)
        st.session_state.messages.append({"role": "assistant", "content": output_text})

# Display action buttons
if len(st.session_state["messages"]) > 0:
    action_buttons_container = st.container()
    action_buttons_container.float(
        "bottom: 6.9rem;background-color: var(--default-backgroundColor); padding-top: 1rem;"
    )

    # Define column dimensions for action buttons
    cols_dimensions = [7, 14.9, 14.5, 9.1, 9, 8.6, 8.7]
    cols_dimensions.append(100 - sum(cols_dimensions))

    col0, col1, col2, col3, col4, col5, col6, col7 = action_buttons_container.columns(
        cols_dimensions
    )

    with col1:
        json_messages = json.dumps(st.session_state["messages"]).encode("utf-8")
        st.download_button(
            label="📥 Save!",
            data=json_messages,
            file_name="chat_conversation.json",
            mime="application/json",
        )

    with col2:
        if st.button("Clear 🧹"):
            st.session_state["messages"] = []
            st.rerun()

    with col4:
        icon = "🔁"
        if st.button(icon):
            st.session_state["rerun"] = True
            st.rerun()

    with col5:
        icon = "👍"
        if st.button(icon):
            log_feedback(icon)

    with col6:
        icon = "👎"
        if st.button(icon):
            log_feedback(icon)

    with col7:
        enc = tiktoken.get_encoding("cl100k_base")
        tokenized_full_text = enc.encode(
            " ".join([item["content"] for item in st.session_state["messages"]])
        )
        label = f"💬 {len(tokenized_full_text)} tokens"
        st.link_button(label, "https://platform.openai.com/tokenizer")

else:
    if "disclaimer" not in st.session_state:
        with st.empty():
            for seconds in range(3):
                st.warning(
                    "‎ You can click on 👍 or 👎 to provide feedback regarding the quality of responses.",
                    icon="💡",
                )
                time.sleep(1)
            st.write("")
            st.session_state["disclaimer"] = True

import streamlit as st
from rag_chatbot import handle_user_input
import datetime as dt
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.messages import AIMessage, HumanMessage


logo = "https://github.com/j0ntendo/med-rag-chatbot/blob/main/chatbot_logo.jpg?raw=true"
st.title("DentaReAct Chatbot")
st.sidebar.image(logo, width=150, use_column_width=True)
st.sidebar.write("Welcome to DentaReAct! Ask me anything dental related and I will do my best to help you.")


if 'generated_history' not in st.session_state:
    st.session_state['generated_history'] = []
if 'user_history' not in st.session_state:
    st.session_state['user_history'] = []
if 'stored_sessions' not in st.session_state:
    st.session_state['stored_sessions'] = []
if 'input' not in st.session_state:
    st.session_state['input'] = ''



if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=10,
        return_messages=True
    )

#init chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

#display chatmessage
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# ---------------- Chatbot ----------------



if prompt := st.chat_input("Ask me anything dental related"):
    with st.chat_message("user"):
        st.markdown(prompt) 
    st.session_state.messages.append({"role": "user", "content": prompt})


if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            prompt = st.session_state.messages[-1]["content"]
    
            response = handle_user_input(prompt)
            output_text = response['output']
            st.markdown(output_text)
            
            # save to history
            message = {"role": "assistant", "content": output_text}
            st.session_state.messages.append(message)


# ---------------- history ----------------
import datetime as dt

def cache_chat():
    """Save current chat history as a stored session."""
    record = []
    for user_msg, ai_msg in zip(st.session_state['user_history'], st.session_state['generated_history']):
        record.append(f'User: {user_msg}')
        record.append(f'AI: {ai_msg}')
    record.append(f'\nSession saved on: {dt.datetime.now().strftime("%m/%d/%Y, %H:%M")}')
    st.session_state['stored_sessions'].append('\n'.join(record))

def reset_session():
    """Clear the chat history and chatbot memory."""
    st.session_state['generated_history'] = []
    st.session_state['user_history'] = []
    if 'memory' in st.session_state:
        st.session_state.memory.clear()

def clear_stored_sessions():
    """Clear the list of stored chat sessions."""
    st.session_state['stored_sessions'] = []


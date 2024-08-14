import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatOpenAI as OpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from chatbot_tools.retriever import bm25_retrieval, mmr_retrieval, ddg_retrieval
from chatbot_tools.prompt import get_react_template
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import openai
import streamlit as st


load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


llm = OpenAI(
    model_name="gpt-4o-mini",
    
)

tools = [
    bm25_retrieval, 
    ddg_retrieval,
    mmr_retrieval
]


prompt_template = get_react_template()

react_agent = create_react_agent(llm, tools, prompt_template)

memory = ConversationBufferWindowMemory(k=6, return_messages=True)
agent_executor = AgentExecutor(
    agent=react_agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    memory=memory
)


def handle_user_input(user_input: str) -> str:
    
    result = agent_executor.invoke({"input": user_input})
    
    return result



# import os
# from dotenv import load_dotenv
# from langchain import hub
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain_community.chat_models import ChatOpenAI as OpenAI
# from langchain_community.tools.tavily_search import TavilySearchResults
# from chatbot_tools.retriever import bm25_retrieval, mmr_retrieval, ddg_retrieval
# from chatbot_tools.prompt import get_react_template
# import streamlit as st

# load_dotenv()


# llm = OpenAI(
#     model_name="gpt-4o-mini"
#     # temperature=0.7,
#     # api_key=os.getenv("OPENAI_API_KEY")
# )


# tools = [
#     bm25_retrieval, 
#     ddg_retrieval,
#     mmr_retrieval
#     ]


# prompt = get_react_template()


# react_agent = create_react_agent(llm, tools, prompt)


# agent_executor = AgentExecutor(agent=react_agent, 
#                                tools=tools, 
#                                verbose=True,
#                                handle_parsing_errors=True)


# user_input = "can I drink cola after wisdom teeth removal?"


# result = agent_executor.invoke({"input": user_input})


# print(result)

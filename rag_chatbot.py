import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from chatbot_tools.retriever import bm25_retrieval, mmr_retrieval
from chatbot_tools.prompt import get_react_template

load_dotenv()


model = OpenAI(
    model_name="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
    )


tools = [bm25_retrieval, mmr_retrieval]


prompt = get_react_template()


react_agent = create_react_agent(llm=model, tools=tools, prompt=prompt)


agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)


user_input = "can I drink cola after wisdom teeth removal?"


result = agent_executor.invoke({"input": user_input})


print(result)



if __name__ == "__main__":
    user_input = "What are some tips for studying efficiently?"
    result = agent_executor.invoke({"input": user_input})
    print(result)

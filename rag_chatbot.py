import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from tools.retriever import retrieve_bm25, retrieve_mmr
from tools.prompt.react_template import get_react_template
os.environ['OPENAI_API_KEY'] = ""

load_dotenv()


model = OpenAI(model_name="gpt-4o-mini")


tools = [retrieve_bm25, retrieve_mmr]


prompt = get_react_template()


react_agent = create_react_agent(model=model, tools=tools, prompt=prompt)


agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)


user_input = "can I drink cola after wisdom teeth removal?"


result = agent_executor.invoke({"input": user_input})


print(result)



if __name__ == "__main__":
    user_input = "What are some tips for studying efficiently?"
    result = agent_executor.invoke({"input": user_input})
    print(result)


from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.agents import tool
import os
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults

os.environ['OPENAI_API_KEY'] = ""

load_dotenv()

def get_react_template():
    return PromptTemplate.from_template("""You are a chatbot, answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

Begin!

Question: {input}

Thought:{agent_scratchpad}
                                        """)

def initialize_vector_store():
    """
    Initializes and returns a Chroma vector store for the 'dental_restoration_data' collection.

    Returns:
        Chroma: The initialized Chroma vector store.
    """
    embedding_function = OpenAIEmbeddings()  # Specify the embedding functio
    return Chroma(
        collection_name="dental_restoration_data",
        persist_directory="./chroma_langchain_db",
        embedding_function=embedding_function,
    )


# @tool
# def bm25_retrieval(query: str):
#     """
#     Retrieves relevant documents from the Chroma vector store using the BM25 algorithm.

#     Args:
#         query (str): The search query used to find relevant documents.

#     Returns:
#         List[Dict[str, str]]: A list of dictionaries where each dictionary contains the title and dialogue of the retrieved documents.
#     """
#     vector_store = initialize_vector_store()

    
#     # Fetch all documents
#     documents = vector_store.get_by_ids("ids")
#     # Convert raw documents to Document objects if needed
#     #documents = [Document(page_content=doc['page_content'], metadata=doc['metadata']) for doc in raw_documents['documents']]

    
#     bm25_retriever = BM25Retriever.from_documents(documents)

    
#     results = bm25_retriever.get_relevant_documents(query)

    
#     formatted_results = [
#         {"title": doc.metadata['title'], "dialogue": doc.page_content} for doc in results
#     ]
#     return formatted_results

@tool
def mmr_retrieval(query: str):
    """
    Retrieves relevant documents from the Chroma vector store using the Maximal Marginal Relevance (MMR) algorithm.

    Args:
        query (str): The search query used to find relevant documents.

    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary contains the title and dialogue of the retrieved documents.
    """
    vector_store = initialize_vector_store()

    
    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 20, 'lambda_mult': 0.25}
    )

    
    results = mmr_retriever.get_relevant_documents(query)

    
    formatted_results = [
        {"title": doc.metadata['title'], "dialogue": doc.page_content} for doc in results
    ]
    return formatted_results

@tool
def ddg_retrieval(query: str):
    """
    Performs a search using DuckDuckGo and formats the results to include snippets and links.

    Args:
        query (str): The search query to perform.

    Returns:
        List[Dict[str, str]]: A list of dictionaries with snippet and link for each result.
    """
    search = DuckDuckGoSearchResults(
        description="A tool that queries DuckDuckGo to retrieve search results specific to dentistry. Use this tool to find up-to-date information on dental care, procedures, and oral health. Input should be a search query related to dentistry. The output is a list of search results, each including a snippet summarizing the information and a link to the full article or source.",
        max_results=5
    )
    results = search.invoke(query)
    
    return results
    



llm = OpenAI()


tools = [ddg_retrieval]


prompt = get_react_template()


react_agent = create_react_agent(llm, tools, prompt)


agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)


user_input = "can I drink cola after wisdom teeth removal?"


result = agent_executor.invoke({"input": user_input})


print(result)

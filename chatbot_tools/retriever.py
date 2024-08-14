import os

from dotenv import load_dotenv

from langchain_community.vectorstores.chroma import Chroma
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.agents import tool

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
# from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI as OpenAI
# from langchain_openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults

load_dotenv()

# Shared persist directory
persist_directory = "./chroma_langchain_db"

# List of collection names
collection_names = [
    "dental_restoration_data",
    "genden_data",
    "oralsurgery_data",
    "orthodontics_data",
    "patient_data",
    "periodontics_data"
]

def initialize_vector_store(collection_name):
    """
    Initializes and returns a Chroma vector store for the given collection in the shared persist directory.

    Args:
        collection_name (str): The name of the collection.

    Returns:
        Chroma: The initialized Chroma vector store.
    """
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")  # Specify the embedding function
    
    collection_persist_directory = os.path.join(persist_directory, collection_name)
    
    return Chroma(
        collection_name=collection_name,
        persist_directory=collection_persist_directory,
        embedding_function=embedding_function,
    )


@tool
def bm25_retrieval(query: str):
    """
    Retrieves relevant documents using BM25 algorithm.

    Args:
        query (str): The search query used to find relevant documents.

    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary contains the title and dialogue of the retrieved documents.
    """
    all_results = []

    for collection_name in collection_names:
        vector_store = initialize_vector_store(collection_name)
        
        # Fetch all documents
        documents = vector_store.get()
        
        text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                      chunk_overlap=0,
                                                      length_function=len)

        documents = text_spliter.create_documents(texts=documents["documents"],
                                                  metadatas=documents["metadatas"])
        
        bm25_retriever = BM25Retriever.from_documents(documents)
        results = bm25_retriever.get_relevant_documents(query)
        
        formatted_results = [
            {"collection": collection_name, "title": doc.metadata['title'], "dialogue": doc.page_content} 
            for doc in results
        ]
        all_results.extend(formatted_results)

    return all_results

@tool
def mmr_retrieval(query: str):
    """
    Retrieves relevant documents using the Maximal Marginal Relevance (MMR) algorithm.

    Args:
        query (str): The search query used to find relevant documents.

    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary contains the title and dialogue of the retrieved documents.
    """
    all_results = []

    for collection_name in collection_names:
        vector_store = initialize_vector_store(collection_name)
        
        mmr_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 3, 'fetch_k': 20, 'lambda_mult': 0.25}
        )

        results = mmr_retriever.get_relevant_documents(query)
        
        formatted_results = [
            {"collection": collection_name, "title": doc.metadata['title'], "dialogue": doc.page_content} 
            for doc in results
        ]
        all_results.extend(formatted_results)

    return all_results


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
    




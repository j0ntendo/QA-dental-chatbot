
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.agents import tool


def initialize_vector_store():
    """
    Initializes and returns a Chroma vector store for the 'dental_restoration_data' collection.

    Returns:
        Chroma: The initialized Chroma vector store.
    """
    return Chroma(
        collection_name="dental_restoration_data",
        persist_directory="./chroma_langchain_db",
    )


@tool
def bm25_retrieval(query: str):
    """
    Retrieves relevant documents from the Chroma vector store using the BM25 algorithm.

    Args:
        query (str): The search query used to find relevant documents.

    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary contains the title and dialogue of the retrieved documents.
    """
    vector_store = initialize_vector_store()

    
    documents = vector_store.get()

    
    bm25_retriever = BM25Retriever.from_documents(documents)

    
    results = bm25_retriever.get_relevant_documents(query)

    
    formatted_results = [
        {"title": doc.metadata['title'], "dialogue": doc.page_content} for doc in results
    ]
    return formatted_results


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
        search_kwargs={'k': 5, 'fetch_k': 20, 'lambda_mult': 0.25}
    )

    
    results = mmr_retriever.get_relevant_documents(query)

    
    formatted_results = [
        {"title": doc.metadata['title'], "dialogue": doc.page_content} for doc in results
    ]
    return formatted_results

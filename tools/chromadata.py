# retrieval.py
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.agents import tool

# Initialize ChromaDB
def initialize_vector_store():
    return Chroma(
        collection_name="dental_restoration_data",
        persist_directory="./chroma_langchain_db",
    )

# BM25 Retriever Tool
@tool
def bm25_retrieval(query: str):
    vector_store = initialize_vector_store()

    # Fetch all documents for BM25 retrieval
    documents = vector_store.get()
    
    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)

    # Retrieve documents based on the title
    results = bm25_retriever.get_relevant_documents(query)
    
    # Format the output
    formatted_results = [
        {"title": doc.metadata['title'], "dialogue": doc.page_content} for doc in results
    ]
    return formatted_results

# Conversational Retrieval Chain Tool
@tool
def conversational_retrieval(query: str):
    vector_store = initialize_vector_store()

    # Initialize the language model
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Create a Conversational Retrieval Chain using RAG
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_store.as_retriever(),
        verbose=True
    )

    # Get response from the ReAct agent
    result = retrieval_chain({"query": query})
    
    return result['output']

# For testing purposes
if __name__ == "__main__":
    # Test BM25 retrieval
    bm25_results = bm25_retrieval("Gum disease")
    print("BM25 Retrieval Results:")
    for result in bm25_results:
        print(f"Title: {result['title']}\nDialogue: {result['dialogue']}\n")

    # Test Conversational retrieval
    conv_result = conversational_retrieval("How should I deal with gum disease caused by grinding?")
    print("Conversational Retrieval Result:")
    print(conv_result)

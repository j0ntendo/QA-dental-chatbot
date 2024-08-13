import os
from dotenv import load_dotenv
import json
import logging
from uuid import uuid4
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
load_dotenv()

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def create_documents(data):
    """Create Document objects for each entry in the dataset."""
    documents = []
    for item in data:
        doc_id = str(uuid4())
        dialogue_doc = Document(
            page_content=item['dialogue'],  
            metadata={"id": item['id'], "title": item['title']},  
            id=doc_id
        )
        documents.append(dialogue_doc)
    return documents

def embed_and_store(documents):
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    
    vector_store = Chroma(
        collection_name="periodontics_data",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
        # metadata={"hnsw:space": "cosine"}
    )

    
    vector_store.add_documents(documents=documents)
    vector_store.persist()  

    print("Documents have been embedded and stored in ChromaDB.")

def main():
    
    json_data = load_json('DATA/periodontics_data.json')

    
    documents = create_documents(json_data)

    
    embed_and_store(documents)

if __name__ == "__main__":
    main()

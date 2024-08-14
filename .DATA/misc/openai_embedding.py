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

def embed_and_store(collection_name, documents):
    """Embed the documents and store them in ChromaDB."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    persist_directory = f"./chroma_langchain_db/{collection_name}"
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    vector_store.add_documents(documents=documents)
    vector_store.persist()

    print(f"Documents for {collection_name} have been embedded and stored in ChromaDB.")

def process_json_files(json_files):
    """Process multiple JSON files and store them in separate ChromaDB collections."""
    for file_path in json_files:
        collection_name = os.path.splitext(os.path.basename(file_path))[0]
        
        json_data = load_json(file_path)
        
        documents = create_documents(json_data)
        
        embed_and_store(collection_name, documents)

def main():
    json_files = [
        '.DATA/dental_restoration_data.json',
        '.DATA/genden_data.json',
        '.DATA/oralsurgery_data.json',
        '.DATA/orthodontics_data.json',
        '.DATA/patient_data.json',
        '.DATA/periodontics_data.json'
    ]
    
    process_json_files(json_files)

if __name__ == "__main__":
    main()

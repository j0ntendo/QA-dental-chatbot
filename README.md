# Dental Chatbot

Welcome to the DentaReAct repository!

<img width="758" alt="Screenshot 2024-08-30 at 10 37 18 PM" src="https://github.com/user-attachments/assets/3e161bc5-3c06-4f1d-ba1f-a542460f4855">

## Features

- **React Agent**: Integrated for enhanced conversational capabilities.
- **RAG (Retrieval-Augmented Generation)**: Used for generating responses based on retrieved information.
- **Streamlit**: Provides an interactive web interface.
- **ChromaDB**: For managing and querying the database.

## Getting Started

### Prerequisites

1. **Personal OpenAI API Key**: You will need a personal OpenAI API key to run the chatbot.

2. **Libraries**: Install the required libraries by running:
   pip install -r requirements.txt
   
3. **Running the Chatbot**
To start the chatbot interface using Streamlit, run the following command:
python -m streamlit run main.py --server.port 4900

4. **Tools Used for React Agent**
BM25 Retriever: A probabilistic-based retrieval method.
MMR Retriever: A method to diversify the retrieved documents.
DuckDuckGo Retriever: For querying web data.

**Database**
The database for the chatbot was crawled from Dentistry Forums.

**legal**
This project was done during the 2024 summer internship at DenComm.


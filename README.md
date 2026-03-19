# Barbados Licensing Authority RAG Chatbot

This project is an AI-powered chatbot that uses a Retrieval-Augmented Generation (RAG) pipeline with adaptive late chunking to answer questions based on Barbados Licensing Authority (BLA) documents.

The system ingests structured JSON data, converts it into vector embeddings, stores it in a Qdrant vector database, and serves responses through a Flask API using an LLM via OpenRouter.

---

## Project Structure

├── app3.py          # Flask API + RAG query pipeline

├── ingest.py        # Data ingestion and vector indexing

├── data/            # Folder containing JSON documents

├── .env             # Environment variables (API keys)

├── requirements.txt # Dependencies

---

## Features

- Retrieval-Augmented Generation (RAG) architecture  
- Adaptive late chunking for improved retrieval quality  
- Semantic search using sentence embeddings  
- Keyword-based reranking for better relevance  
- Qdrant vector database integration  
- OpenRouter LLM (Qwen3-32B) for response generation  
- Flask API backend with CORS support  
- Local frontend support via HTTP server  

---

## Data Source

The data used for this project was sourced from publicly accessible information from the Barbados Licensing Authority (bla.gov.bb).

---

## Setup Instructions

### 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-name>
---

### 2. Create and activate a virtual environment
python -m venv venv

**Windows:**
venv\Scripts\activate

**Mac/Linux:**
source venv/bin/activate

---

### 3. Install dependencies
pip install -r requirements.txt

---

### 4. Configure environment variables

Create a `.env` file in the root directory:
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

---

### 5. Prepare data

Create a `data` folder and place your JSON files inside:
data/
├── file1.json
├── file2.json

---

## Running the System

### Step 1: Ingest data into Qdrant
python ingest.py

This will:
- Extract text from JSON files  
- Generate embeddings  
- Apply adaptive late chunking  
- Upload vectors to Qdrant  

---

### Step 2: Start the backend API
python app3.py

The API will run on:
http://localhost:8000

---

### Step 3: Run frontend (static server)
python -m http.server 3000

Open in browser:
http://localhost:3000/chatbot.html

---

## How It Works

### Ingestion Pipeline (`ingest.py`)

1. Loads JSON documents from the `data` folder  
2. Extracts text using a flexible parser  
3. Generates full-document embeddings  
4. Applies adaptive late chunking:  
   - Chunk size varies based on document length  
   - Embeddings are weighted based on chunk proportion  
5. Stores both:  
   - Full document vectors (baseline)  
   - Chunk-level vectors (for retrieval)  
6. Uploads data to Qdrant with batching and retry logic  

---

### Query Pipeline (`app3.py`)

1. User submits a query via `/chat`  
2. Query is converted into an embedding  
3. Retrieval occurs in two stages:  
   - Broad semantic search (top 30 results)  
   - Keyword-based reranking (top 10 refined results)  
4. Context is constructed from top chunks  
5. LLM generates a response using OpenRouter (Qwen3-32B)  
6. Response is returned as JSON  

---

## API Endpoints

### `GET /`

Health check endpoint showing:
- API status  
- Number of indexed vectors  

---

### `POST /chat`

**Request:**
                          {
"query": "How do I renew my driver's license?"
}
                          
**Response:**
                          {
"answer": "..."
}
                          
---

## Key Design Decisions

- Late Chunking improves context preservation compared to naive chunking  
- Adaptive chunk sizes handle both short and long documents efficiently  
- Hybrid retrieval combines semantic search with keyword matching  
- Storing both full documents and chunks allows flexible retrieval  
- Low temperature LLM ensures factual and consistent responses  

---

## Requirements

Create a `requirements.txt` file with:
flask
flask-cors
python-dotenv
qdrant-client
sentence-transformers
transformers
torch
openai

---

## Notes

- Ensure your Qdrant collection name matches in both scripts (`rag_latechunk_test2`)  
- The ingestion script deletes and recreates the collection each run  
- JSON files should contain meaningful text content for best results  
- The system is optimized for structured government/service documents  

---

## Future Improvements

- Add hybrid search (BM25 + vector search)  
- Implement caching for faster responses  
- Add authentication and rate limiting  
- Deploy using Docker or cloud services  
- Improve frontend UI/UX  

                          

                          
                          




























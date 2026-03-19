from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()  # Loads OPENROUTER_API_KEY, QDRANT_URL, QDRANT_API_KEY, etc.

# Create Flask app instance
app = Flask(__name__)

# Enable CORS for frontend running on localhost:3000
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])


# -----------------------------
# Initialize RAG Components
# -----------------------------

# Connect to Qdrant vector database using URL + API key from environment
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Load embedding model used to convert text into vectors
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Name of Qdrant collection that stores your document chunks
collection_name = "rag_latechunk_test2"


# -----------------------------
# OpenRouter (Qwen3‑32B) Client
# -----------------------------

openrouter_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_key:
    raise ValueError("Missing OPENROUTER_API_KEY in .env file")

openrouter_client = OpenAI(
    api_key=openrouter_key,
    base_url="https://openrouter.ai/api/v1"
)

print("OpenRouter client ready (Qwen3‑32B endpoint)")


# -----------------------------
# Home Route (Health Check)
# -----------------------------
@app.route('/')
def home():
    # Get total number of vectors stored in the collection
    count = client.get_collection(collection_name).points_count

    # Return simple HTML status page
    return f"""
    <h1> Barbados Licensing Authority Chatbot API</h1>
    <p>Backend running with <strong>{count}</strong> vectors indexed</p>
    <p> Frontend: <a href="http://localhost:3000/chatbot.html">Open Chatbot UI</a></p>
    """


# -----------------------------
# Chat Endpoint (Main RAG Logic)
# -----------------------------
@app.route('/chat', methods=['GET', 'POST', 'OPTIONS'])
def chat():
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    # If accessed via browser GET request
    if request.method == 'GET':
        count = client.get_collection(collection_name).points_count
        return jsonify({'answer': f'BLA Assistant ready! {count} documents indexed.'})

    # Extract user query from JSON body
    query = request.json.get('query', '').strip()

    # Validate empty input
    if not query:
        return jsonify({'answer': 'Please enter a question about BLA services.'}), 400

    print(f" Searching for: '{query}'")

    # ==================== HIGH ACCURACY RETRIEVAL ====================

    # Convert user query into embedding vector
    query_vector = model.encode([query])[0].tolist()

    # -----------------------------
    # Pass 1: Broad Semantic Search
    # -----------------------------
    # Retrieve top 30 semantically similar chunks
    broad_results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=30,  # Retrieve more candidates for reranking
        search_params={
            "hnsw_ef": 128  # Controls search accuracy vs speed
        }
    )

    # -----------------------------
    # Pass 2: Keyword Reranking
    # -----------------------------
    # Break user query into individual words
    query_words = set(query.lower().split())  # Use set for fast lookup

    # Sort top 20 results by:
    # 1. Number of keyword matches
    # 2. Length of chunk (prefer richer content)
    reranked = sorted(
        broad_results.points[:20],
        key=lambda p: (
            sum(1 for word in query_words if word in p.payload['text'].lower()),
            len(p.payload['text'])
        ),
        reverse=True
    )[:10]  # Keep final top 10 chunks

    # Combine selected chunks into context block
    # Each chunk truncated to 1000 characters to avoid token overflow
    context = "\n\n---\n".join([p.payload['text'][:1000] for p in reranked])

    print(f" Found {len(reranked)} optimized chunks")

    # ==================== ENHANCED PROMPT ====================

    prompt = f"""BLA Assistant - Answer using ONLY these Barbados Licensing Authority documents.

CONTEXT ({len(reranked)} relevant chunks found):
{context}

USER QUERY: {query}

RULES:
1. Be SPECIFIC with forms, emails, websites, steps
2. Be concise but give details
3. Use your OWN WORDS - never copy text directly
4. Say "I found related info but need more details" if unclear
5. Make users sound CAPABLE: "You can easily..."


ACTIONABLE ANSWER:"""

    # ==================== OpenRouter (Qwen3‑32B) Generation ====================
    try:
        response = openrouter_client.chat.completions.create(
            model="qwen/qwen3-32b",   # OpenRouter model string for Qwen3‑32B
            messages=[
                {"role": "system", "content": "You are a concise Barbados Licensing Authority expert."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=800,
            temperature=0.1,
        )

        answer = response.choices[0].message.content.strip()
        print(f" OpenRouter (Qwen3‑32B) Success ({len(answer)} chars)")

    except Exception as e:
        print(f" OpenRouter Error: {e}")
        answer = "You can easily contact [BLAsupport@barbados.gov.bb](mailto:BLAsupport@barbados.gov.bb) or visit bla.gov.bb for assistance."

    # Return JSON response to frontend
    return jsonify({'answer': answer})


# -----------------------------
# Run Application
# -----------------------------
if __name__ == '__main__':
    # Print startup information
    count = client.get_collection(collection_name).points_count
    print(f" BLA Chatbot API (OpenRouter + RAG) v2.0...")
    print(f" {count} vectors indexed")
    print(f" Frontend: http://localhost:3000/chatbot.html")

    # Start Flask server
    app.run(host='0.0.0.0', port=8000, debug=True)

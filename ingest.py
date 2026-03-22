import os
import json
import uuid
import time
import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# -----------------------------
# Connect to Qdrant Cloud
# -----------------------------
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
print(" Connected to Qdrant cluster")

# -----------------------------
# Setup embedding model + tokenizer
# -----------------------------
model_name = "sentence-transformers/all-MiniLM-L6-v2" #dense
sentence_model = SentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(" Model & tokenizer loaded")

# -----------------------------
# Collection setup
# -----------------------------
collection_name = "rag_latechunk_test2"
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
    print(f"  Deleted existing collection '{collection_name}'")
    
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
print(f" Collection '{collection_name}' ready")

# -----------------------------
# SAFE JSON TEXT EXTRACTION (No clean_text dependency)
# -----------------------------
def extract_text_safely(doc, filename):
    """Extract text from ANY JSON structure"""
    if isinstance(doc, dict):
        # Try multiple common paths
        paths = ["content.text", "content", "text", "content_clean", "full_text"]
        for path in paths:
            keys = path.split('.')
            value = doc
            try:
                for key in keys:
                    value = value[key]
                if isinstance(value, str) and len(value.strip()) > 50:
                    print(f"   Found text in '{path}' ({len(value)} chars)")
                    return value
            except (KeyError, TypeError):
                continue
        
        # Fallback: find largest strings
        all_texts = []
        def find_text(obj):
            if isinstance(obj, str) and len(obj.strip()) > 30:
                all_texts.append(obj.strip())
            elif isinstance(obj, dict):
                for v in obj.values():
                    find_text(v)
            elif isinstance(obj, list):
                for item in obj:
                    find_text(item)
        find_text(doc)
        
        if all_texts:
            full_text = " ".join(all_texts[:3])
            print(f"   Using fallback texts ({len(full_text)} chars)")
            return full_text
    
    return str(doc)[:5000]

# -----------------------------
# ADAPTIVE Late Chunking (Varying document lengths)
# -----------------------------
def adaptive_late_chunk(text, base_chunk_size=256, base_overlap=32):
    """Adaptive Late Chunking: Optimizes for short/medium/long docs"""
    full_embedding = sentence_model.encode([text])[0]  
    
    # ADAPTIVE SIZING
    doc_length = len(text)
    if doc_length < 500:
        chunk_size, overlap = 128, 20
    elif doc_length < 2000:
        chunk_size, overlap = 256, 32
    elif doc_length < 5000:
        chunk_size, overlap = 384, 50
    else:
        chunk_size, overlap = 512, 75
    
    print(f"   Doc {doc_length:,} chars → chunks={chunk_size}, overlap={overlap}")
    
    if len(text) < chunk_size:
        return [text], [full_embedding]
    
    chunks = []
    chunk_embeddings = []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
            
            # Late chunking: Weight by chunk proportion
            chunk_weight = len(chunk) / len(text)
            weighted_embedding = full_embedding * chunk_weight
            chunk_embeddings.append(weighted_embedding)
    
    return chunks, chunk_embeddings

# -----------------------------
# Data ingestion loop
# -----------------------------
DATA_FOLDER = "data"
points = []
total_full_docs = 0
total_chunks = 0

for file in os.listdir(DATA_FOLDER):
    if file.endswith(".json"):
        print(f"\n Processing: {file}")
        try:
            with open(os.path.join(DATA_FOLDER, file), "r", encoding="utf-8") as f:
                doc = json.load(f)
                
                # Safe extraction
                full_text = extract_text_safely(doc, file)
                if len(full_text.strip()) < 50:
                    print(f"   Skipping {file} - too little text")
                    continue
                
                # Safe metadata
                title = doc.get("metadata", {}).get("doc_title", file)
                doc_id = doc.get("metadata", {}).get("document_id", str(uuid.uuid4()))

                # 1. FULL DOCUMENT (baseline)
                full_uuid = str(uuid.uuid4())
                full_embedding = sentence_model.encode([full_text])[0].tolist()
                points.append(PointStruct(
                    id=full_uuid,
                    vector=full_embedding,
                    payload={
                        "title": title,
                        "text": full_text,
                        "source": file,
                        "type": "full",
                        "document_id": doc_id,
                        "strategy": "baseline"
                    }
                ))
                total_full_docs += 1

                # 2. ADAPTIVE LATE CHUNKS
                chunks, chunk_embeddings = adaptive_late_chunk(full_text)
                for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    chunk_uuid = str(uuid.uuid4())
                    vector_list = embedding.tolist()
                    
                    assert len(vector_list) == 384, f"Vector dim mismatch: {len(vector_list)}"
                    
                    points.append(PointStruct(
                        id=chunk_uuid,
                        vector=vector_list,
                        payload={
                            "title": title,
                            "text": chunk,
                            "source": file,
                            "type": "late_chunk",
                            "document_id": doc_id,
                            "chunk_index": i,
                            "strategy": "adaptive_late_chunking"
                        }
                    ))
                total_chunks += len(chunks)
                print(f"   1 full doc + {len(chunks)} adaptive chunks")
                
        except Exception as e:
            print(f"   Error processing {file}: {e}")
            continue

print("\n" + "="*60)
print(" UPLOADING TO QDRANT (with rate limiting)...")
print("="*60)

# -----------------------------
# FIXED: Reliable upload with retries + rate limiting
# -----------------------------
BATCH_SIZE = 25      # Smaller batches
MAX_RETRIES = 3      # Retry failed batches

for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i+BATCH_SIZE]
    
    for attempt in range(MAX_RETRIES):
        try:
            client.upsert(collection_name=collection_name, points=batch)
            print(f" Batch {i//BATCH_SIZE + 1} ({len(batch)} points) ")
            break  # Success - exit retry loop
            
        except Exception as e:
            wait_time = (2 ** attempt) + 1  # 1s, 3s, 7s backoff
            print(f"    Attempt {attempt+1} failed: {str(e)[:50]}... waiting {wait_time}s")
            
            if attempt == MAX_RETRIES - 1:
                print(f"   Batch {i//BATCH_SIZE + 1} FINAL FAILURE")
                continue
            time.sleep(wait_time)  # Wait before retry
    
    time.sleep(0.5)  # 0.5s pause between ALL batches

print(f"\n INGESTION COMPLETE!")
print(f"   STATS:")
print(f"   Full documents:     {total_full_docs}")
print(f"   Adaptive chunks:    {total_chunks}")
print(f"   Total vectors:      {len(points)}")

# Final verification
try:
    collection_info = client.get_collection(collection_name)
    print(f" Collection '{collection_name}': {collection_info.points_count} points ready for RAG!")
except:
    print("  Could not verify collection - check Qdrant dashboard")

print("\n Your BLA RAG database is LIVE!")

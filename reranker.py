import os
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Load models
print("Loading models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chromadb_store")
collection = client.get_or_create_collection(name="docs")

# Get all chunks
all_data = collection.get()
all_chunks = all_data["documents"]

print(f"Loaded {len(all_chunks)} chunks from ChromaDB")

# Your question
question = "What is the main methodology described in this document?"

# ---- VECTOR SEARCH ----
print("\nRunning vector search...")
question_embedding = embedder.encode(question).tolist()
vector_results = collection.query(
    query_embeddings=[question_embedding],
    n_results=5
)
vector_chunks = vector_results["documents"][0]

# ---- BM25 SEARCH ----
print("Running BM25 search...")
tokenized_chunks = [chunk.split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)
bm25_scores = bm25.get_scores(question.split())
top_bm25_indices = np.argsort(bm25_scores)[::-1][:5]
bm25_chunks = [all_chunks[i] for i in top_bm25_indices]

# ---- COMBINE ----
seen = set()
combined_chunks = []
for chunk in vector_chunks + bm25_chunks:
    if chunk not in seen:
        seen.add(chunk)
        combined_chunks.append(chunk)

print(f"Combined chunks before reranking: {len(combined_chunks)}")

# ---- RERANK ----
print("Reranking chunks...")
pairs = [[question, chunk] for chunk in combined_chunks]
scores = reranker.predict(pairs)

# Sort chunks by reranker score
scored_chunks = list(zip(scores, combined_chunks))
scored_chunks.sort(key=lambda x: x[0], reverse=True)

# Take top 3 after reranking
top_chunks = [chunk for score, chunk in scored_chunks[:3]]

print("\nTop 3 chunks after reranking:")
for i, (score, chunk) in enumerate(scored_chunks[:3]):
    print(f"\n--- Chunk {i+1} (score: {score:.4f}) ---")
    print(chunk[:200])

# ---- SEND TO GROQ ----
context = "\n\n".join(top_chunks)

print("\n\nSending to Groq...\n")
response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": """You are a helpful assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say 'I don't have enough information to answer this.'
Always end your answer with: 'Source: ai methodology.pdf'"""
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
)

print(f"Question: {question}\n")
print(f"Answer: {response.choices[0].message.content}")
import os
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Connect to ChromaDB and get all chunks
client = chromadb.PersistentClient(path="./chromadb_store")
collection = client.get_or_create_collection(name="docs")

# Get all stored chunks
all_data = collection.get()
all_chunks = all_data["documents"]
all_ids = all_data["ids"]

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

# ---- BM25 KEYWORD SEARCH ----
print("Running BM25 search...")
tokenized_chunks = [chunk.split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)
tokenized_question = question.split()
bm25_scores = bm25.get_scores(tokenized_question)

# Get top 5 BM25 results
import numpy as np
top_bm25_indices = np.argsort(bm25_scores)[::-1][:5]
bm25_chunks = [all_chunks[i] for i in top_bm25_indices]

# ---- COMBINE RESULTS ----
print("Combining results...")
seen = set()
combined_chunks = []

for chunk in vector_chunks + bm25_chunks:
    if chunk not in seen:
        seen.add(chunk)
        combined_chunks.append(chunk)

# Take top 5 unique chunks
final_chunks = combined_chunks[:5]
context = "\n\n".join(final_chunks)

print(f"Final chunks after hybrid merge: {len(final_chunks)}")

# ---- SEND TO GROQ ----
print("\nSending to Groq...\n")
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
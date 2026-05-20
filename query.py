import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("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"))

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chromadb_store")
collection = client.get_or_create_collection(name="docs")

# Your question
question = "What is the main methodology described in this document?"

# Embed the question and search
question_embedding = embedder.encode(question).tolist()
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

# Build context from retrieved chunks
chunks = results["documents"][0]
context = "\n\n".join(chunks)

# Send to Groq with context
print("Sending to Groq...\n")
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
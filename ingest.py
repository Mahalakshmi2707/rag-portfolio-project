import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer

# Step 1: Read the PDF
print("Reading PDF...")
reader = PdfReader("docs/ai methodology.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

print(f"Total characters read: {len(full_text)}")

# Step 2: Split into chunks
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(full_text)
print(f"Total chunks created: {len(chunks)}")

# Step 3: Load embedding model
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Store in ChromaDB
print("Storing in ChromaDB...")
client = chromadb.PersistentClient(path="./chromadb_store")
collection = client.get_or_create_collection(name="docs")

for i, chunk in enumerate(chunks):
    embedding = embedder.encode(chunk).tolist()
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[f"chunk_{i}"]
    )

print(f"Done! {len(chunks)} chunks stored in ChromaDB.")
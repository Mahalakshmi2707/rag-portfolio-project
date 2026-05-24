import os
import chromadb
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def auto_ingest():
    """Auto ingest PDF on startup if ChromaDB is empty"""
    collection = load_chromadb()
    existing = collection.get()
    
    if len(existing["ids"]) > 0:
        return  # already ingested
    
    pdf_path = "docs/ai methodology.pdf"
    if not os.path.exists(pdf_path):
        return  # no PDF found
    
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    print("Auto ingesting document...")
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(full_text)
    embedder_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    for i, chunk in enumerate(chunks):
        embedding = embedder_model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk_{i}"]
        )
    print(f"Auto ingestion complete — {len(chunks)} chunks stored.")

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="centered"
)

# ---- CUSTOM CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: #0d0d0d;
        color: #f0f0f0;
    }

    /* Header */
    .header-container {
        text-align: center;
        padding: 3rem 0 1rem 0;
    }
    .header-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7c6fff, #48cfad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .header-sub {
        color: #666;
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    /* Pipeline tags */
    .tags {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1.5rem 0 2.5rem 0;
    }
    .tag {
        background: rgba(124, 111, 255, 0.1);
        border: 1px solid rgba(124, 111, 255, 0.3);
        color: #9d91ff;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    /* Input */
    .stTextInput > div > div > input {
        background: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 14px !important;
        color: #f0f0f0 !important;
        padding: 0.9rem 1.2rem !important;
        font-size: 1rem !important;
        transition: border 0.2s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #7c6fff !important;
        box-shadow: 0 0 0 3px rgba(124, 111, 255, 0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #444 !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #7c6fff, #48cfad) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        letter-spacing: 0.02em !important;
        transition: opacity 0.2s, transform 0.1s !important;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, rgba(72,207,173,0.07), rgba(124,111,255,0.07));
        border: 1px solid rgba(72, 207, 173, 0.25);
        border-radius: 18px;
        padding: 1.8rem;
        margin: 2rem 0 1rem 0;
    }
    .answer-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #48cfad;
        margin-bottom: 0.8rem;
    }
    .answer-text {
        font-size: 1rem;
        color: #e8e8e8;
        line-height: 1.8;
    }

    /* Metrics */
    .metrics-row {
        display: flex;
        gap: 1rem;
        margin: 1.2rem 0;
    }
    .metric-box {
        flex: 1;
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #7c6fff;
    }
    .metric-label {
        font-size: 0.68rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.2rem;
    }

    /* Chunk cards */
    .chunk-card {
        background: #141414;
        border: 1px solid #222;
        border-left: 3px solid #7c6fff;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.8rem;
        font-size: 0.86rem;
        color: #999;
        line-height: 1.65;
    }
    .chunk-header {
        font-size: 0.68rem;
        font-weight: 700;
        color: #7c6fff;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    /* History item */
    .history-item {
        background: #141414;
        border: 1px solid #1f1f1f;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .history-q {
        font-size: 0.82rem;
        color: #7c6fff;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .history-a {
        font-size: 0.85rem;
        color: #888;
        line-height: 1.6;
    }

    /* Divider */
    hr { border-color: #1f1f1f !important; }

    /* Hide streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("""
<div class="header-container">
    <div class="header-title">🧠 DocMind AI</div>
    <div class="header-sub">Production-grade document intelligence · Hybrid RAG pipeline</div>
</div>
<div class="tags">
    <span class="tag">📄 PDF Ingestion</span>
    <span class="tag">🔍 Vector Search</span>
    <span class="tag">⚡ BM25 Keyword Search</span>
    <span class="tag">🎯 Cross-Encoder Reranking</span>
    <span class="tag">🤖 LLaMA 3.1</span>
    <span class="tag">✅ Citation Enforced</span>
</div>
""", unsafe_allow_html=True)

# ---- LOAD MODELS ----
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return embedder, reranker, groq_client

@st.cache_resource
def load_chromadb():
    client = chromadb.PersistentClient(path="./chromadb_store")
    collection = client.get_or_create_collection(name="docs")
    return collection

embedder, reranker, groq_client = load_models()
collection = load_chromadb()
auto_ingest()

# ---- RETRIEVAL FUNCTION ----
def retrieve_and_answer(question):
    # Vector search
    question_embedding = embedder.encode(question).tolist()
    vector_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )
    vector_chunks = vector_results["documents"][0]

    # BM25 search
    # BM25 search
    all_data = collection.get()
    all_chunks = all_data["documents"]

    if not all_chunks:
        return "No documents found. Please wait for ingestion to complete.", [], [], 0

    tokenized_chunks = [chunk.split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    # Combine
    seen = set()
    combined_chunks = []
    for chunk in vector_chunks + bm25_chunks:
        if chunk not in seen:
            seen.add(chunk)
            combined_chunks.append(chunk)

    # Rerank
    pairs = [[question, chunk] for chunk in combined_chunks]
    scores = reranker.predict(pairs)
    scored_chunks = list(zip(scores, combined_chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for score, chunk in scored_chunks[:3]]
    top_scores = [round(float(score), 4) for score, chunk in scored_chunks[:3]]

    # Send to Groq
    context = "\n\n".join([chunk[:400] for chunk in top_chunks])
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

    return response.choices[0].message.content, top_chunks, top_scores, len(combined_chunks)

# ---- SESSION STATE ----
if "history" not in st.session_state:
    st.session_state.history = []

# ---- INPUT ----
st.markdown("### Ask a Question")
question = st.text_input(
    label="q",
    placeholder="e.g. What is the five-step methodology described in this document?",
    label_visibility="collapsed"
)

ask = st.button("🔍 Search & Answer")

if ask:
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running hybrid retrieval + reranking..."):
            answer, chunks, scores, total = retrieve_and_answer(question)

        # Save to history
        st.session_state.history.append({
            "question": question,
            "answer": answer
        })

        # Metrics
        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-box">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Candidates Retrieved</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">3</div>
                <div class="metric-label">After Reranking</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{scores[0]}</div>
                <div class="metric-label">Top Relevance Score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Answer
        st.markdown(f"""
        <div class="answer-box">
            <div class="answer-label">✅ Answer</div>
            <div class="answer-text">{answer}</div>
        </div>
        """, unsafe_allow_html=True)

        # Chunks
        st.markdown("### 📚 Retrieved Context")
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            st.markdown(f"""
            <div class="chunk-card">
                <div class="chunk-header">Chunk {i+1} · Relevance Score: {score}</div>
                {chunk[:400]}
            </div>
            """, unsafe_allow_html=True)

# ---- HISTORY ----
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕘 Question History")
    for item in reversed(st.session_state.history[-5:]):
        st.markdown(f"""
        <div class="history-item">
            <div class="history-q">Q: {item['question']}</div>
            <div class="history-a">A: {item['answer'][:200]}...</div>
        </div>
        """, unsafe_allow_html=True)
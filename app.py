import os
import chromadb
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide"
)

# ---- CUSTOM CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: #0d0b14;
        color: #f0eeff;
    }

    section[data-testid="stSidebar"] {
        background: #110e1a !important;
        border-right: 1px solid #2a1f3d !important;
    }

    /* Header */
    .header-container { text-align: center; padding: 2rem 0 1rem 0; }
    .header-title {
        font-size: 3.2rem; font-weight: 800;
        background: linear-gradient(135deg, #a855f7, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .header-sub { color: #6b5f8a; font-size: 1rem; margin-top: 0.4rem; }

    /* Tags */
    .tags {
        display: flex; justify-content: center;
        flex-wrap: wrap; gap: 0.5rem; margin: 1.5rem 0 2rem 0;
    }
    .tag {
        background: rgba(168, 85, 247, 0.1);
        border: 1px solid rgba(168, 85, 247, 0.3);
        color: #c084fc; padding: 0.3rem 0.9rem;
        border-radius: 999px; font-size: 0.72rem;
        font-weight: 600; letter-spacing: 0.04em;
    }

    /* Input */
    .stTextInput > div > div > input {
        background: #1a1425 !important;
        border: 1.5px solid #2a1f3d !important;
        border-radius: 14px !important;
        color: #f0eeff !important;
        padding: 0.9rem 1.2rem !important;
        font-size: 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.15) !important;
    }
    .stTextInput > div > div > input::placeholder { color: #4a3f5c !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #a855f7, #ec4899) !important;
        color: #fff !important; border: none !important;
        border-radius: 14px !important; padding: 0.7rem 2rem !important;
        font-weight: 700 !important; font-size: 0.95rem !important;
        width: 100% !important;
        box-shadow: 0 4px 20px rgba(168, 85, 247, 0.35) !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 30px rgba(168, 85, 247, 0.5) !important;
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, rgba(168,85,247,0.08), rgba(236,72,153,0.08));
        border: 1.5px solid rgba(168, 85, 247, 0.3);
        border-radius: 18px; padding: 1.8rem; margin: 2rem 0 1rem 0;
        box-shadow: 0 4px 30px rgba(168, 85, 247, 0.12),
                    inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .answer-label {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: #a855f7; margin-bottom: 0.8rem;
    }
    .answer-text { font-size: 1rem; color: #e8e0ff; line-height: 1.8; }

    /* Metrics */
    .metrics-row { display: flex; gap: 1rem; margin: 1.2rem 0; }
    .metric-box {
        flex: 1;
        background: linear-gradient(135deg, rgba(168,85,247,0.08), rgba(236,72,153,0.05));
        border: 1.5px solid rgba(168, 85, 247, 0.2);
        border-radius: 14px; padding: 1rem; text-align: center;
        box-shadow: 0 2px 12px rgba(168, 85, 247, 0.1);
    }
    .metric-value { font-size: 1.6rem; font-weight: 800; color: #c084fc; }
    .metric-label {
        font-size: 0.68rem; color: #6b5f8a;
        text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem;
    }

    /* Chunk cards */
    .chunk-card {
        background: #1a1425;
        border: 1.5px solid #2a1f3d;
        border-left: 4px solid #a855f7;
        border-radius: 12px; padding: 1.1rem 1.3rem;
        margin-bottom: 0.8rem; font-size: 0.86rem;
        color: #9d8fbe; line-height: 1.65;
        box-shadow: 0 2px 12px rgba(168, 85, 247, 0.08);
    }
    .chunk-header {
        font-size: 0.68rem; font-weight: 700; color: #a855f7;
        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;
    }

    /* History */
    .history-item {
        background: #1a1425;
        border: 1.5px solid #2a1f3d;
        border-radius: 12px; padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 2px 12px rgba(168, 85, 247, 0.06);
    }
    .history-q { font-size: 0.82rem; color: #a855f7; font-weight: 600; margin-bottom: 0.3rem; }
    .history-a { font-size: 0.85rem; color: #6b5f8a; line-height: 1.6; }

    hr { border-color: #2a1f3d !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #1a1425 !important;
        border: 2px dashed rgba(168, 85, 247, 0.4) !important;
        border-radius: 14px !important;
        padding: 1rem !important;
    }

    /* Success/info */
    .stSuccess {
        background: rgba(168,85,247,0.1) !important;
        border-color: #a855f7 !important;
        color: #c084fc !important;
    }
    .stInfo {
        background: rgba(168,85,247,0.08) !important;
        border-color: rgba(168,85,247,0.3) !important;
        color: #c084fc !important;
    }
    .stWarning {
        background: rgba(236,72,153,0.08) !important;
        border-color: rgba(236,72,153,0.3) !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---- LOAD MODELS ----
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return embedder, reranker, groq_client

embedder, reranker, groq_client = load_models()

# ---- IN-MEMORY CHROMADB ----
def get_fresh_collection():
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="docs")
    return collection

# ---- INGEST FUNCTION ----
def ingest_pdf(uploaded_file, collection):
    reader = PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(full_text)

    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk_{i}"]
        )
    return len(chunks)

# ---- RETRIEVAL FUNCTION ----
def retrieve_and_answer(question, collection, filename):
    # Vector search
    question_embedding = embedder.encode(question).tolist()
    vector_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )
    vector_chunks = vector_results["documents"][0]

    # BM25 search
    all_data = collection.get()
    all_chunks = all_data["documents"]

    if not all_chunks:
        return "No documents found. Please upload a PDF first.", [], [], 0

    tokenized_chunks = [chunk.split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(question.split())
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:5]
    bm25_chunks = [all_chunks[i] for i in top_bm25_indices]

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
                "content": f"""You are a helpful assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say 'I don't have enough information to answer this.'
Always end your answer with: 'Source: {filename}'"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    return response.choices[0].message.content, top_chunks, top_scores, len(combined_chunks)

# ---- SESSION STATE ----
if "collection" not in st.session_state:
    st.session_state.collection = get_fresh_collection()
if "history" not in st.session_state:
    st.session_state.history = []
if "filename" not in st.session_state:
    st.session_state.filename = None
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("## 📁 Upload Document")
    st.caption("Upload any PDF to start asking questions.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf"
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.filename:
            with st.spinner("Processing document..."):
                st.session_state.collection = get_fresh_collection()
                st.session_state.history = []
                count = ingest_pdf(uploaded_file, st.session_state.collection)
                st.session_state.filename = uploaded_file.name
                st.session_state.chunks_count = count
            st.success(f"✅ Ready! {count} chunks indexed.")

    st.markdown("---")

    if st.session_state.filename:
        st.markdown("### 📄 Active Document")
        st.markdown(f"`{st.session_state.filename}`")
        st.caption(f"{st.session_state.chunks_count} chunks indexed")
    else:
        st.caption("No document uploaded yet.")

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline")
    st.caption("🔍 Vector Search (semantic)")
    st.caption("⚡ BM25 (keyword)")
    st.caption("🎯 Cross-Encoder Reranking")
    st.caption("🤖 LLaMA 3.1 via Groq")
    st.caption("✅ Citation Enforced")

# ---- MAIN AREA ----
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

if not st.session_state.filename:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    st.markdown("### 💬 Ask a Question")
    question = st.text_input(
        label="q",
        placeholder=f"Ask anything about {st.session_state.filename}...",
        label_visibility="collapsed"
    )

    ask = st.button("🔍 Search & Answer")

    if ask:
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Running hybrid retrieval + reranking..."):
                answer, chunks, scores, total = retrieve_and_answer(
                    question,
                    st.session_state.collection,
                    st.session_state.filename
                )

            st.session_state.history.append({
                "question": question,
                "answer": answer
            })

            top_score = scores[0] if scores else "N/A"

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
                    <div class="metric-value">{top_score}</div>
                    <div class="metric-label">Top Relevance Score</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="answer-box">
                <div class="answer-label">✅ Answer</div>
                <div class="answer-text">{answer}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📚 Retrieved Context")
            for i, (chunk, score) in enumerate(zip(chunks, scores)):
                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-header">Chunk {i+1} · Relevance Score: {score}</div>
                    {chunk[:400]}
                </div>
                """, unsafe_allow_html=True)

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
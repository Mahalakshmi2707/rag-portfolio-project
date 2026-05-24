---
title: DocMind AI
emoji: 🧠
colorFrom: purple
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---
# 🧠 DocMind AI — Production RAG Application

A production-grade **Retrieval Augmented Generation (RAG)** system that answers questions about documents with cited sources. Built with hybrid retrieval, cross-encoder reranking, and citation enforcement.

---

## 🚀 Live Demo
👉 [Try it live on Hugging Face](https://huggingface.co/spaces/maha-1234/docmind-ai)

## 🏗️ Architecture


---

## ⚙️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| LLM | LLaMA 3.1 8B via Groq |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| UI | Streamlit |

---

## 🔍 Pipeline Details

### Hybrid Retrieval
Combines two search strategies for better coverage:
- **Vector Search** — finds semantically similar chunks using cosine similarity
- **BM25** — finds exact keyword matches using probabilistic ranking

### Cross-Encoder Reranking
After hybrid retrieval returns candidate chunks, a cross-encoder model scores each chunk individually against the question and reorders by relevance. This gives significantly more accurate context than retrieval alone.

### Citation Enforcement
The LLM is prompted to answer strictly from retrieved context and always cite the source document. If the answer isn't in the context, it says so rather than hallucinating.

---

## 🛠️ Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/Mahalakshmi2707/rag-portfolio-project.git
cd rag-portfolio-project
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file:

GROQ_API_KEY=your_groq_api_key_here


### 5. Add your PDF
Place your PDF inside the `docs/` folder.

### 6. Ingest your document
```bash
python ingest.py
```

### 7. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

rag-portfolio-project/
│
├── docs/                  # Source documents
├── chromadb_store/        # Vector database (auto-generated)
├── app.py                 # Streamlit UI
├── ingest.py              # Document ingestion pipeline
├── reranker.py            # Retrieval + reranking script
├── hybrid_search.py       # Hybrid search script
├── requirements.txt       # Dependencies
├── .env                   # API keys (never commit)
└── README.md              # This file

---

## 💡 Key Learnings

- How RAG solves the limitation of LLMs on private/custom data
- Why hybrid search outperforms pure vector search
- How cross-encoder reranking improves retrieval precision
- How to enforce citation and prevent hallucination
- Building production AI pipelines with proper project structure

---

## 👩‍💻 Author
**Mahalakshmi** — Aspiring AI/LLM Engineer
=======
---
title: Docmind Ai
emoji: 📚
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 6.14.0
python_version: '3.13'
app_file: app.py
pinned: false
license: mit
short_description: 'A document based questioning application '
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 942625306d463c4dd8d2b081495c0f23e97f8921



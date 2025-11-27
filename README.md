📄 Multi-PDF Retrieval-Augmented Generation (RAG) Chatbot

A production-grade RAG system with multi-PDF ingestion, FAISS vector search, Groq LLMs, and Streamlit chat UI.

<p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge"/> <img src="https://img.shields.io/badge/LangChain-RAG-green?style=for-the-badge"/> <img src="https://img.shields.io/badge/FAISS-Vector%20Search-orange?style=for-the-badge"/> <img src="https://img.shields.io/badge/Groq-LLM-red?style=for-the-badge"/> <img src="https://img.shields.io/badge/Streamlit-UI-pink?style=for-the-badge"/> </p>
🚀 Overview

This project implements a full end-to-end RAG pipeline, capable of:

Ingesting multiple PDFs

Extracting content + preserving metadata (file name, page number)

Chunking text into semantic pieces

Generating vector embeddings using MiniLM L6-V2

Storing & searching vectors using FAISS

Running conversational question-answering over documents using Groq LLMs

Providing a modern, chat-style UI using Streamlit

This is not a simple toy project — the architecture mirrors real production RAG systems used in industry.

🧠 Features
🔹 1. Multi-PDF Ingestion Pipeline

Batch loads all PDFs in data_pdfs/

Extracts text + metadata via PyPDFLoader

Splits text using RecursiveCharacterTextSplitter

🔹 2. FAISS Vector Store

MiniLM embeddings (fast + accurate)

Saves search index locally (faiss_index/)

Metadata-aware retrieval (score, source, page)

🔹 3. Conversational RAG

Maintains chat history

Builds structured prompts with:

retrieved context blocks

last N conversation turns

Query → Retrieve → Generate workflow

🔹 4. Streamlit Chat UI

Chat-style interface

Sidebar controls (top_k, history length)

Persistent session state

Live Groq responses

🏗️ Project Structure
📁 multi-pdf-rag-chatbot/
│
├── a.py                 # PDF ingestion → FAISS index builder
├── b.py                 # CLI chat RAG interface
├── app.py               # Streamlit chat UI
├── .gitignore
├── requirements.txt
├── README.md
│
├── data_pdfs/           # (local only) User PDFs to ingest
└── faiss_index/         # (local only) Generated vector index

⚙️ Setup & Installation
1️⃣ Clone the repository
git clone https://github.com/dhruvvs/Multi-Pdf-RAG-chatbot
cd Multi-Pdf-RAG-chatbot

2️⃣ Create virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your Groq API key

Create a .env file (not tracked by git):

GROQ_API_KEY=your_api_key_here

5️⃣ Add PDFs

Place any PDFs you want to chat with inside:

data_pdfs/

▶️ Running the Application
A. Build the FAISS index

(only required whenever PDFs change)

python a.py

B. Run CLI Chatbot
python b.py

C. Run Streamlit Web UI
streamlit run app.py


Open in browser:
👉 http://localhost:8501

🧩 How It Works (Architecture)
                ┌─────────────────────────────┐
                │         PDF Documents        │
                └──────────────┬──────────────┘
                               │
                     (PyPDFLoader + Chunking)
                               │
                ┌──────────────▼──────────────┐
                │     MiniLM Embeddings        │
                └──────────────┬──────────────┘
                               │
                       (FAISS Index)
                               │
                ┌──────────────▼──────────────┐
                │     Vector Search (Top-K)    │
                └──────────────┬──────────────┘
                               │
                      (Context Assembly)
                               │
                ┌──────────────▼──────────────┐
                │   Groq LLM (LLaMA-3.1 8B)    │
                └──────────────┬──────────────┘
                               │
                         (Final Answer)
                               │
                ┌──────────────▼──────────────┐
                │     CLI / Streamlit UI       │
                └──────────────────────────────┘

📚 Tech Stack

Python 3.10+

LangChain (text splitters, loaders)

FAISS-CPU (vector search)

HuggingFace Sentence Transformers (MiniLM embeddings)

Groq LLM API

Streamlit (web UI)

python-dotenv (secret management)

⭐ Use Cases

Personalized course chatbot

Enterprise knowledge base Q/A

Legal document analysis

Research paper summarization

Multi-document semantic search

AI-powered study assistant

📝 Future Enhancements

“Upload PDF” button in Streamlit

Automatic index refresh

Hybrid search (BM25 + FAISS)

Highlight citations in answers

RAG evaluation (RAGAS / MRR / Recall@k)

Summaries + key topics per PDF

💼 Perfect for Your Resume

This project demonstrates:

LLM orchestration

Embeddings / vector databases

Full-stack ML app development

Modular Python engineering

Prompt engineering

Groq API integration

Real RAG architecture

You can confidently add this as a major GenAI project in your resume or portfolio.

🤝 Contributions

PRs and issues are welcome — feel free to open a discussion.

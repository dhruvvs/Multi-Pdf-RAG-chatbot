# 📘 Multi-PDF Retrieval-Augmented Generation (RAG) Chatbot  
A powerful, modular system that enables conversational question-answering over multiple PDFs using FAISS vector search, MiniLM embeddings, Groq LLMs, and a Streamlit interface.

---

# 📘 Overview  
This project demonstrates how to build a complete **Retrieval-Augmented Generation (RAG)** pipeline capable of reading multiple PDF documents, converting them into vector embeddings, and using a Large Language Model (LLM) to generate grounded answers.

It showcases:

- Automated PDF ingestion  
- Document chunking and metadata preservation  
- Semantic vector search with FAISS  
- Conversational RAG (with chat memory)  
- CLI & Streamlit UI interfaces  

This makes the project ideal for **GenAI applications, enterprise search, document Q/A systems, and educational exploration of retrieval-based LLM workflows**.

---

# 🚀 Objectives  

- Load and process multiple PDFs automatically  
- Split documents into meaningful text chunks with metadata  
- Generate MiniLM sentence embeddings for all chunks  
- Build a persistent FAISS vector index  
- Retrieve top-k relevant document passages  
- Integrate Groq LLMs for fast, grounded Q/A  
- Provide both terminal and web-based chat interfaces  
- Maintain multi-turn conversation history  

---

# 🛠️ Tech Stack  

| Category       | Tools / Libraries |
|----------------|------------------|
| Language       | Python |
| Embeddings     | SentenceTransformers (MiniLM L6-V2) |
| LLM Framework  | Groq API (LLaMA-3.1 / Mixtral / etc.) |
| Vector Search  | FAISS (CPU) |
| Document Loaders | LangChain PyPDFLoader |
| Chunking       | LangChain RecursiveCharacterTextSplitter |
| UI             | Streamlit |
| Development    | VS Code, `.env` environment variables |

---

# 🧾 Features  

### ✔ Multi-PDF Ingestion  
Load all PDFs in the `/data_pdfs/` directory with automatic metadata assignment (file name, page number).

### ✔ Text Chunking with Metadata  
Documents are split into overlapping chunks to maximize retrieval accuracy.

### ✔ FAISS Vector Store  
Efficient similarity search with saved index persistence.

### ✔ Conversational RAG  
Model responds using ONLY retrieved document context, with multi-turn history.

### ✔ Streamlit Web Interface  
Chat-style interface with:

- History tracking  
- Top-k tuning  
- Real-time responses  
- Sidebar controls  

### ✔ CLI Chatbot  
Lightweight command-line interface to interact with the RAG engine.

---

# 📊 System Capabilities  

### ✔ PDF-Aware Semantic Search  
Retrieve the most relevant passages from your corpus using cosine similarity on dense vector embeddings.

### ✔ Context-Grounded Q/A  
The LLM is constrained to answer *only* from retrieved document content.

### ✔ Conversation Memory  
Multi-turn reasoning simulates a chat experience while referring back to previous questions.

### ✔ Real-Time Streamlit Chat  
A clean and modern interface to explore your entire document collection interactively.

---

# 📦 Visual / Functional Components  

### ✔ Retrieval Flow  
- PDF → Text → Chunks → Embeddings → FAISS  
- Query → Top-k search → Context assembly → LLM answer  

### ✔ Context Blocks  
Each answer (internally) is based on structured retrieved sections with:

- Rank  
- Source (filename)  
- Page number  
- Similarity score  

### ✔ Streamlit Interface  
- Chat window  
- Sidebar controls  
- Auto-scrolled message layout  
- Clear chat option  

---

# ▶ How It Works  

### 1. **PDF Ingestion (`a.py`)**  
Reads all PDFs from the directory, extracts text, chunks content, embeds, and builds FAISS index.

### 2. **RAG Core (`b.py`)**  
Runs query → top-k retrieval → Groq LLM → answer.  
Maintains conversation history for coherence.

### 3. **Front-End UI (`app.py`)**  
Streamlit chat interface for real-time document Q/A.

---

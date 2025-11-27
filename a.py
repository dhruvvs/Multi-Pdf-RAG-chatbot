# a.py  -> build FAISS index from all PDFs in data_pdfs/

import os
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

PDF_DIR = "data_pdfs"     # folder where your PDFs live
INDEX_DIR = "faiss_index" # folder where FAISS index will be stored
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def collect_pdf_paths(pdf_dir: str) -> List[str]:
    """Return list of absolute paths to all PDFs in pdf_dir."""
    paths: List[str] = []
    os.makedirs(pdf_dir, exist_ok=True)
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            paths.append(os.path.join(pdf_dir, fname))
    return sorted(paths)


def load_all_pdfs(pdf_paths: List[str]):
    """Load all PDFs into LangChain Document objects with metadata."""
    # ✅ Compatible with new & old LangChain versions
    try:
        from langchain_core.documents import Document  # new path
    except ImportError:
        from langchain.schema import Document  # fallback for older versions

    all_docs: List[Document] = []

    for pdf_path in pdf_paths:
        print(f"[INGEST] Loading: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        for i, d in enumerate(docs):
            d.metadata = d.metadata or {}
            d.metadata["source"] = os.path.basename(pdf_path)
            # page index (0-based), fallback to i
            d.metadata["page"] = d.metadata.get("page", i)
            all_docs.append(d)

    print(f"[INGEST] Total pages loaded: {len(all_docs)}")
    return all_docs


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        add_start_index=True,
    )


def build_vectorstore_from_pdfs(
    pdf_dir: str = PDF_DIR,
    index_dir: str = INDEX_DIR,
) -> FAISS:
    pdf_paths = collect_pdf_paths(pdf_dir)
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in '{pdf_dir}'. Put your PDFs there and run again."
        )

    all_docs = load_all_pdfs(pdf_paths)

    print("[SPLIT] Splitting documents into chunks...")
    splitter = build_text_splitter()
    chunks = splitter.split_documents(all_docs)
    print(f"[SPLIT] Total chunks: {len(chunks)}")

    print("[EMBED] Building embeddings + FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"[SAVE] FAISS index saved to '{index_dir}'")

    return vectorstore


if __name__ == "__main__":
    print("=== RAG Ingestion: Start ===")
    build_vectorstore_from_pdfs()
    print("=== RAG Ingestion: Done ===")

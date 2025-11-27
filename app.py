# app.py  -> Streamlit UI for RAG chatbot

import os

import streamlit as st # type: ignore
from dotenv import load_dotenv # type: ignore
from groq import Groq # type: ignore

from langchain_community.vectorstores import FAISS # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore

from b import RAGChatSession, INDEX_DIR, EMBEDDING_MODEL_NAME, GROQ_MODEL # type: ignore

load_dotenv()


@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if not os.path.isdir(INDEX_DIR):
        raise FileNotFoundError(
            f"Index directory '{INDEX_DIR}' not found. Run a.py first."
        )
    vs = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    return vs


@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env/environment")
    return Groq(api_key=api_key)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # list[{"role","content"}]


def main():
    st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="wide")
    st.title("📄 RAG PDF Chatbot")
    st.caption("Ask questions over your ingested PDFs")

    init_session_state()

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top-k chunks", 2, 10, 5)
        max_hist = st.slider("History turns", 0, 10, 6)

        if st.button("Clear chat history"):
            st.session_state["messages"].clear()
            st.experimental_rerun()

        st.markdown("---")
        st.markdown("Index directory:")
        st.code(INDEX_DIR)

    vectorstore = load_vectorstore()
    groq_client = get_groq_client()

    session = RAGChatSession(
        vectorstore=vectorstore,
        groq_client=groq_client,
        top_k=top_k,
        max_history_turns=max_hist,
    )
    session.chat_history = list(st.session_state["messages"])

    # render previous messages
    for msg in st.session_state["messages"]:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask something about your PDFs")
    if user_q:
        st.session_state["messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        answer = session.ask(user_q)

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    main()

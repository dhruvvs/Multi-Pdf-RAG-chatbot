# b.py  -> CLI RAG chatbot

import os
from typing import List, Dict, Tuple

from dotenv import load_dotenv # type: ignore
from groq import Groq # type: ignore

from langchain_community.vectorstores import FAISS # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore

# Same config as a.py
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 5
MAX_HISTORY_TURNS = 6  # number of previous Q/A turns to keep


load_dotenv()


def load_vectorstore(index_dir: str = INDEX_DIR) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(
            f"Index directory '{index_dir}' not found. Run a.py first."
        )
    vs = FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )
    return vs


class RAGChatSession:
    """Simple conversational RAG session with history."""

    def __init__(
        self,
        vectorstore: FAISS,
        groq_client: Groq,
        top_k: int = TOP_K,
        max_history_turns: int = MAX_HISTORY_TURNS,
    ):
        self.vectorstore = vectorstore
        self.groq = groq_client
        self.top_k = top_k
        self.max_history_turns = max_history_turns
        self.chat_history: List[Dict[str, str]] = []

    def _retrieve(self, question: str):
        return self.vectorstore.similarity_search_with_score(question, k=self.top_k)

    def _build_context_block(self, docs_and_scores) -> str:
        blocks = []
        for rank, (doc, score) in enumerate(docs_and_scores, start=1):
            meta = doc.metadata or {}
            source = meta.get("source", "unknown.pdf")
            page = meta.get("page", "?")
            blocks.append(
                f"[Rank {rank} | Source: {source} | Page: {page} | Score: {score:.4f}]\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(blocks)

    def _build_history_block(self) -> str:
        if not self.chat_history:
            return "No previous conversation.\n"
        recent = self.chat_history[-2 * self.max_history_turns :]
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def _build_prompt(self, question: str) -> str:
        docs_and_scores = self._retrieve(question)
        context_block = self._build_context_block(docs_and_scores)
        history_block = self._build_history_block()

        prompt = f"""
You are a Retrieval-Augmented Generation (RAG) assistant answering ONLY from the provided context.
If the answer is not clearly in the context, reply:
"I don't know based on the provided documents."

--- Context (Top-{self.top_k} retrieved chunks) ---
{context_block}

--- Conversation History ---
{history_block}

--- User Question ---
{question}

Now answer:
"""
        return prompt

    def ask(self, question: str) -> str:
        prompt = self._build_prompt(question)

        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
        )

        answer = response.choices[0].message.content.strip()

        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})

        return answer


def main_cli():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment or .env")

    groq_client = Groq(api_key=api_key)
    vectorstore = load_vectorstore()
    session = RAGChatSession(vectorstore, groq_client)

    print("=== RAG PDF Chatbot (CLI) ===")
    print("Ask questions about your PDFs. Type 'exit' to quit.\n")

    while True:
        try:
            user_q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Bye]")
            break

        if user_q.lower() in {"exit", "quit"}:
            print("Assistant: Goodbye!")
            break
        if not user_q:
            continue

        answer = session.ask(user_q)
        print("\nAssistant:\n", answer, "\n")


if __name__ == "__main__":
    main_cli()

"""Intelligent Document Assistant
Upload a document (PDF, DOCX, TXT), then summarize it or ask questions about it.
"""
import os
import re
from collections import Counter
from math import log

import streamlit as st
from anthropic import Anthropic
from docx import Document
from pypdf import PdfReader

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
CHUNK_SIZE = 1200      # characters per chunk
TOP_K = 4              # chunks sent to the model per question
MAX_SUMMARY_CHARS = 60000


# ---------- 1. Document loading ----------
def load_document(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    return file.read().decode("utf-8", errors="ignore")


# ---------- 2. Chunking and simple retrieval (TF-IDF style) ----------
def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) > size and current:
            chunks.append(current)
            current = ""
        current += p + "\n"
    if current:
        chunks.append(current)
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieve(question: str, chunks: list[str], k: int = TOP_K) -> list[str]:
    q_tokens = set(tokenize(question))
    docs = [Counter(tokenize(c)) for c in chunks]
    n = len(chunks)
    scores = []
    for i, counts in enumerate(docs):
        score = 0.0
        for t in q_tokens:
            if counts[t]:
                df = sum(1 for d in docs if d[t])
                score += counts[t] * log(1 + n / df)
        scores.append((score, i))
    top = sorted(scores, reverse=True)[:k]
    return [chunks[i] for s, i in sorted(top, key=lambda x: x[1]) if s > 0] or chunks[:k]


# ---------- 3. Claude calls ----------
def ask_claude(client: Anthropic, system: str, user: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def summarize(client: Anthropic, text: str) -> str:
    return ask_claude(
        client,
        "You are a document assistant. Write a clear, concise summary with key points.",
        f"Summarize this document:\n\n{text[:MAX_SUMMARY_CHARS]}",
    )


def answer(client: Anthropic, question: str, chunks: list[str]) -> str:
    context = "\n---\n".join(retrieve(question, chunks))
    return ask_claude(
        client,
        "Answer using ONLY the provided document excerpts. "
        "If the answer is not in them, say you could not find it.",
        f"Excerpts:\n{context}\n\nQuestion: {question}",
    )


# ---------- 4. UI ----------
st.set_page_config(page_title="Intelligent Document Assistant", page_icon="📄")
st.title("📄 Intelligent Document Assistant")

api_key = os.getenv("ANTHROPIC_API_KEY") or st.sidebar.text_input(
    "Anthropic API key", type="password"
)
uploaded = st.sidebar.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

if not api_key:
    st.info("Enter your Anthropic API key in the sidebar (or set ANTHROPIC_API_KEY).")
    st.stop()
if not uploaded:
    st.info("Upload a PDF, DOCX, or TXT file to begin.")
    st.stop()

client = Anthropic(api_key=api_key)

if st.session_state.get("file_name") != uploaded.name:
    text = load_document(uploaded)
    st.session_state.update(
        file_name=uploaded.name, text=text, chunks=chunk_text(text), history=[]
    )

text, chunks = st.session_state.text, st.session_state.chunks
if not text.strip():
    st.error("No readable text found (scanned PDFs need OCR).")
    st.stop()

st.caption(f"{uploaded.name}: {len(text):,} characters, {len(chunks)} chunks")

tab_sum, tab_qa = st.tabs(["Summary", "Ask questions"])

with tab_sum:
    if st.button("Summarize document"):
        with st.spinner("Summarizing..."):
            st.markdown(summarize(client, text))

with tab_qa:
    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)
    if question := st.chat_input("Ask something about the document"):
        st.chat_message("user").write(question)
        with st.spinner("Thinking..."):
            reply = answer(client, question, chunks)
        st.chat_message("assistant").write(reply)
        st.session_state.history += [("user", question), ("assistant", reply)]

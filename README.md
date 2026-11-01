# Intelligent Document Assistant

Upload a PDF, DOCX or TXT file, get a summary, and ask questions about it.

## How it works
1. **Load**: extracts text from the uploaded file (pypdf / python-docx).
2. **Chunk**: splits the text into ~1200-character pieces.
3. **Retrieve**: scores chunks against your question (TF-IDF style) and picks the top 4.
4. **Answer**: sends only those excerpts to Claude, which answers from them.

## Run
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"     # Windows: set ANTHROPIC_API_KEY=your-key
streamlit run app.py
```

Optional: `CLAUDE_MODEL` environment variable to change the model (default `claude-sonnet-5-5`).

## Ideas to extend
- Replace keyword retrieval with embeddings + a vector store (FAISS / Chroma)
- OCR for scanned PDFs (pytesseract)
- Multiple documents, citations with page numbers, export of chat history

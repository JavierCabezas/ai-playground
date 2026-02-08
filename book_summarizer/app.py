import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from trainer import IndexerWrapper, INDEX_DIR

app = FastAPI()

# Global AI components
qa_model = None
encoder = None
index = None
chunks = []
chunk_metadata = []


class QARequest(BaseModel):
    question: str
    book_name: str


def bootstrap():
    global qa_model, encoder, index, chunks, chunk_metadata

    # Run indexer if files are missing [cite: 1]
    if not (INDEX_DIR / "books.index").exists():
        print("[bootstrap] Index not found. Building now...")
        indexer = IndexerWrapper()
        indexer.run()

    # Load offline-ready models
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(str(INDEX_DIR / "books.index"))

    # Load chunks with book-specific metadata [cite: 2, 3, 4]
    if (INDEX_DIR / "chunks.txt").exists():
        with open(INDEX_DIR / "chunks.txt", "r", encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    fname, text = line.split("|", 1)
                    chunks.append(text.strip())
                    chunk_metadata.append(fname.strip())


bootstrap()


@app.get("/", response_class=HTMLResponse)
async def get_gui():
    # Dynamically list books from the data folder [cite: 1]
    books = [f.name for f in Path("data").glob("*.txt")]
    options = "".join([f'<option value="{b}">{b}</option>' for b in books])

    return f"""
    <html>
        <head>
            <title>Offline Book AI</title>
            <style>
                body {{ font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; }}
                .container {{ background: #f9f9f9; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                select, input, button {{ width: 100%; padding: 12px; margin: 10px 0; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }}
                button {{ background: #007bff; color: white; border: none; cursor: pointer; font-weight: bold; }}
                #response {{ margin-top: 20px; padding: 15px; border-left: 4px solid #007bff; background: white; display: none; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>📖 Offline Book Explorer</h2>
                <p>Select a book and ask a question. Everything is processed inside Docker.</p>

                <label>Choose a Source:</label>
                <select id="bookSelect">{options}</select>

                <label>Your Question:</label>
                <input type="text" id="questionInput" placeholder="What happened in the garden?">

                <button onclick="askQuestion()">Ask AI</button>

                <div id="response">
                    <strong>Answer:</strong> <span id="answerText"></span><br><br>
                    <small style="color: #666;"><strong>Retrieved Context:</strong> <span id="contextText"></span></small>
                </div>
            </div>

            <script>
                async function askQuestion() {{
                    const book = document.getElementById('bookSelect').value;
                    const question = document.getElementById('questionInput').value;
                    const resDiv = document.getElementById('response');

                    resDiv.style.display = "block";
                    document.getElementById('answerText').innerText = "Analyzing book content...";
                    document.getElementById('contextText').innerText = "...";

                    const response = await fetch('/qa', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{ "question": question, "book_name": book }})
                    }});

                    const data = await response.json();
                    document.getElementById('answerText').innerText = data.answer;
                    document.getElementById('contextText').innerText = data.source_used;
                }}
            </script>
        </body>
    </html>
    """


@app.post("/qa")
async def answer(req: QARequest):
    # Semantic Search [cite: 1]
    question_embedding = encoder.encode([req.question])
    distances, indices = index.search(question_embedding.astype('float32'), k=10)

    # Filter retrieved chunks to only the selected book [cite: 1]
    relevant_chunks = []
    for idx in indices[0]:
        if chunk_metadata[idx] == req.book_name:
            relevant_chunks.append(chunks[idx])
        if len(relevant_chunks) >= 2: break

    context = " ".join(relevant_chunks) if relevant_chunks else "I couldn't find specific info in that book."

    # Generate answer from context [cite: 1]
    result = qa_model(question=req.question, context=context)
    return {
        "answer": result["answer"],
        "source_used": context[:300] + "..."
    }

if __name__ == "__main__":
    import uvicorn
    # This keeps the process running so the container doesn't exit
    uvicorn.run(app, host="0.0.0.0", port=8000)
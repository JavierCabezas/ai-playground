import os
import shutil
import numpy as np
import faiss
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from transformers import pipeline
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Configuration
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Global AI components
gen_model = None
encoder = None
current_index = None
current_chunks = []
current_book_name = "None"


@app.on_event("startup")
async def startup_event():
    global gen_model, encoder
    print("🚀 Loading AI models... (This may take a minute on first run)")
    # Generative model for summarization and logic
    gen_model = pipeline("text2text-generation", model="google/flan-t5-base")
    # Semantic encoder for search
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Models loaded.")


@app.get("/", response_class=HTMLResponse)
async def get_gui():
    return f"""
    <html>
        <head>
            <title>Offline Book AI</title>
            <style>
                body {{ font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f8f9fa; }}
                .container {{ background: white; padding: 30px; border-radius: 12px; shadow: 0 4px 12px rgba(0,0,0,0.1); }}
                button {{ padding: 12px; margin: 10px 0; width: 100%; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; }}
                .btn-upload {{ background: #28a745; color: white; }}
                .btn-ask {{ background: #007bff; color: white; }}
                #res {{ margin-top: 20px; padding: 15px; background: #eee; border-radius: 6px; display: none; white-space: pre-wrap; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>📖 Solo-Book Explorer</h2>
                <p>Current: <b id="bn">{current_book_name}</b></p>
                <input type="file" id="f" accept=".txt">
                <button class="btn-upload" onclick="u()">Upload & Index</button>
                <hr>
                <input type="text" id="q" style="width:100%; padding:10px;" placeholder="Ask or Summarize...">
                <button class="btn-ask" onclick="a('qa')">Ask AI</button>
                <button style="background:#6c757d; color:white;" onclick="a('sum')">Summarize Book</button>
                <div id="res"></div>
            </div>
            <script>
                async function u() {{
                    const file = document.getElementById('f').files[0];
                    if(!file) return;
                    const fd = new FormData(); fd.append("file", file);
                    document.getElementById('res').style.display = "block";
                    document.getElementById('res').innerText = "Indexing...";
                    await fetch('/upload', {{ method: 'POST', body: fd }});
                    location.reload();
                }}
                async function a(t) {{
                    const out = document.getElementById('res');
                    out.style.display = "block";
                    out.innerText = "Processing...";
                    const ep = t === 'sum' ? '/summarize' : '/qa';
                    const r = await fetch(ep, {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
                        body: new URLSearchParams({{ 'question': document.getElementById('q').value }})
                    }});
                    const d = await r.json();
                    out.innerHTML = `<b>Result:</b><br>${{d.answer || d.summary}}`;
                }}
            </script>
        </body>
    </html>
    """


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global current_index, current_chunks, current_book_name
    content = (await file.read()).decode("utf-8")
    words = content.split()
    current_chunks = [" ".join(words[i:i + 300]) for i in range(0, len(words), 300)]
    current_book_name = file.filename

    embeddings = encoder.encode(current_chunks)
    current_index = faiss.IndexFlatL2(embeddings.shape[1])
    current_index.add(np.array(embeddings).astype('float32'))
    return {"status": "ok"}


@app.post("/qa")
async def qa(question: str = Form(...)):
    if not current_index: return {"answer": "Upload a book first."}
    q_emb = encoder.encode([question])
    _, i = current_index.search(q_emb.astype('float32'), k=5)
    context = " ".join([current_chunks[idx] for idx in i[0]])
    # Prompt with repetition penalty to fix the "fox, fox" issue
    res = gen_model(f"Answer based on context: {context} Question: {question}", max_length=150, repetition_penalty=2.5)
    return {"answer": res[0]['generated_text']}


@app.post("/summarize")
async def summarize():
    if not current_chunks: return {"summary": "No book."}
    partials = [gen_model(f"summarize: {c}", max_length=50)[0]['generated_text'] for c in current_chunks[:15]]
    res = gen_model("Summarize these points: " + " ".join(partials), max_length=300, repetition_penalty=2.0)
    return {"summary": res[0]['generated_text']}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
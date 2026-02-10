import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from trainer import IndexerWrapper, INDEX_DIR

app = FastAPI()

# Global AI components
gen_model = None  # Generative model for QA and Summarization
encoder = None
index = None
chunks = []
chunk_metadata = []


class QARequest(BaseModel):
    question: str
    book_name: str


def bootstrap():
    global gen_model, encoder, index, chunks, chunk_metadata
    if not (INDEX_DIR / "books.index").exists():
        IndexerWrapper().run()

    # Load Flan-T5-Base (Generative)
    # This model can summarize and answer complex questions
    gen_model = pipeline("text2text-generation", model="google/flan-t5-base")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(str(INDEX_DIR / "books.index"))

    with open(INDEX_DIR / "chunks.txt", "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                fname, text = line.split("|", 1)
                chunks.append(text.strip())
                chunk_metadata.append(fname.strip())


bootstrap()


@app.get("/", response_class=HTMLResponse)
async def get_gui():
    books = [f.name for f in Path("data").glob("*.txt")]
    options = "".join([f'<option value="{b}">{b}</option>' for b in books])
    return f"""
    <html>
        <head>
            <title>Offline Book AI (Generative)</title>
            <style>
                body {{ font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }}
                .box {{ border: 1px solid #ccc; padding: 20px; border-radius: 8px; background: #fdfdfd; }}
                button {{ padding: 10px; cursor: pointer; border-radius: 4px; border: none; font-weight: bold; }}
                .btn-ask {{ background: #007bff; color: white; width: 70%; }}
                .btn-sum {{ background: #28a745; color: white; width: 28%; }}
                #res {{ margin-top: 20px; white-space: pre-wrap; background: #eee; padding: 15px; border-radius: 4px; display: none; }}
            </style>
        </head>
        <body>
            <div class="box">
                <h2>📚 Generative Book Explorer</h2>
                <select id="b" style="width:100%; padding:10px; margin-bottom:10px;">{options}</select>
                <input type="text" id="q" style="width:100%; padding:10px; margin-bottom:10px;" placeholder="Ask a question or request a summary...">

                <button class="btn-ask" onclick="action('qa')">Ask Question</button>
                <button class="btn-sum" onclick="action('sum')">Summarize Book</button>

                <div id="res"></div>
            </div>
            <script>
                async function action(type) {{
                    const out = document.getElementById('res');
                    out.style.display = "block";
                    out.innerText = type === 'sum' ? "Processing full book (Map-Reduce)... this may take 1-2 mins." : "Searching and generating...";

                    const endpoint = type === 'sum' ? '/summarize' : '/qa';
                    const payload = type === 'sum' ? {{ book_name: document.getElementById('b').value }} : {{ question: document.getElementById('q').value, book_name: document.getElementById('b').value }};

                    const r = await fetch(endpoint, {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(payload)
                    }});
                    const d = await r.json();
                    out.innerHTML = `<strong>Result:</strong><br>${{d.answer || d.summary}}`;
                }}
            </script>
        </body>
    </html>
    """


@app.post("/qa")
async def answer(req: QARequest):
    # Retrieve top 5 chunks for more "contextual" awareness
    q_emb = encoder.encode([req.question])
    _, indices = index.search(q_emb.astype('float32'), k=5)

    context = ""
    for idx in indices[0]:
        if chunk_metadata[idx] == req.book_name:
            context += chunks[idx] + " "

    # Better prompting for generative models
    prompt = f"Answer the following question based only on the provided context.\nContext: {context}\nQuestion: {req.question}"
    result = gen_model(prompt, max_length=150)

    return {"answer": result[0]['generated_text']}


@app.post("/summarize")
async def summarize(req: dict):
    book_name = req.get("book_name")
    # 1. Gather all chunks for the book
    book_chunks = [chunks[i] for i, name in enumerate(chunk_metadata) if name == book_name]

    # 2. Map: Summarize each chunk
    partial_summaries = []
    # To save time in the demo, we summarize every 2nd chunk or limit total count
    for chunk in book_chunks[:15]:
        res = gen_model(f"summarize in one sentence: {chunk}", max_length=50)
        partial_summaries.append(res[0]['generated_text'])

    # 3. Reduce: Final synthesis
    final_prompt = "Combine these points into a cohesive summary of the book: " + " ".join(partial_summaries)
    final_res = gen_model(final_prompt, max_length=300)

    return {"summary": final_res[0]['generated_text']}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_DIR = Path("data")
INDEX_DIR = Path("models/index")


class IndexerWrapper:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text):
        """Removes Gutenberg legal headers[cite: 2, 3]."""
        start_marker = "*** START OF THE PROJECT GUTENBERG"
        end_marker = "*** END OF THE PROJECT GUTENBERG"

        s_idx = text.find(start_marker)
        if s_idx != -1:
            text = text[text.find("\n", s_idx) + 1:]

        e_idx = text.find(end_marker)
        if e_idx != -1:
            text = text[:e_idx]
        return text.strip()

    def run(self):
        all_chunks = []
        book_files = list(DATA_DIR.glob("*.txt"))

        for book_path in book_files:
            print(f"Processing {book_path.name}...")
            with open(book_path, 'r', encoding='utf-8') as f:
                content = self.clean_text(f.read())
                # Chunk into 400-word blocks for AI readability
                words = content.split()
                chunks = [" ".join(words[i:i + 400]) for i in range(0, len(words), 400)]
                for c in chunks:
                    all_chunks.append(f"{book_path.name} | {c}")

        if not all_chunks: return

        # Create the searchable vector database
        embeddings = self.encoder.encode([c.split("|")[1] for c in all_chunks])
        dim = embeddings.shape[1]
        idx = faiss.IndexFlatL2(dim)
        idx.add(np.array(embeddings).astype('float32'))

        faiss.write_index(idx, str(INDEX_DIR / "books.index"))
        with open(INDEX_DIR / "chunks.txt", "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(chunk.replace("\n", " ") + "\n")
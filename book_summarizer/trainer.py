import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_DIR = Path("data")
INDEX_DIR = Path("models/index")


class IndexerWrapper:
    def __init__(self):
        # Local model that turns text into numerical meaning
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        all_chunks = []
        book_files = list(DATA_DIR.glob("*.txt"))

        for book_path in book_files:
            print(f"Indexing {book_path.name}...")
            with open(book_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split into chunks so the AI doesn't get overwhelmed
                words = content.split()
                chunks = [" ".join(words[i:i + 300]) for i in range(0, len(words), 300)]
                for c in chunks:
                    # We store the filename so the UI can filter by book
                    all_chunks.append(f"{book_path.name} | {c}")

        if not all_chunks:
            print("No text found in data folder!")
            return

        # Create the searchable vector database
        embeddings = self.encoder.encode([c.split("|")[1] for c in all_chunks])
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))

        # Save everything to the models volume
        faiss.write_index(index, str(INDEX_DIR / "books.index"))
        with open(INDEX_DIR / "chunks.txt", "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(chunk.replace("\n", " ") + "\n")
        print("Indexing complete.")
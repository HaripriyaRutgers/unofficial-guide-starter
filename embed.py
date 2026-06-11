"""
embed.py — Stage 3 of the RAG pipeline: Embedding + Vector Store.

Loads chunks.json, embeds every chunk's text locally with all-MiniLM-L6-v2,
and stores the vectors (+ documents + metadata) in a persistent ChromaDB
collection named "cs_careers".

Safe to re-run: if the collection already has documents, it skips embedding
so you don't get duplicates.

Run with:  python embed.py
Libraries:  sentence-transformers, chromadb, json, time
"""

import sys   # only to read the optional "--reset" command-line flag
import json
import time

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.json"
DB_DIR = "./chroma_db"          # persistent on-disk store (created if missing)
COLLECTION_NAME = "cs_careers"
MODEL_NAME = "all-MiniLM-L6-v2"


def main(reset=False):
    # 1) Load all chunks produced by the chunking stage.
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # 2) Open a PERSISTENT Chroma client. PersistentClient writes to disk at
    #    DB_DIR, so the index survives between runs (unlike the in-memory Client()).
    client = chromadb.PersistentClient(path=DB_DIR)

    # --reset: drop the existing collection so we rebuild from clean chunks.
    # delete_collection wipes the collection's vectors/documents/metadata; the
    # next get_or_create below makes a fresh, empty one.
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"--reset: deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            # Raised when the collection doesn't exist yet — nothing to wipe.
            print(f"--reset: no existing '{COLLECTION_NAME}' collection to delete.")

    # get_or_create_collection: fetch the collection if it exists, else make it.
    # metadata {"hnsw:space": "cosine"} sets the distance metric to COSINE.
    # Chroma defaults to squared-L2, which on these (un-normalized) MiniLM
    # vectors gives large, hard-to-interpret distances. Cosine gives a clean
    # 0..2 scale (0 = identical) that matches the 0.6 threshold guidance below.
    # NOTE: the space is baked in at creation time — to change it you must
    # delete ./chroma_db and re-run, since an existing collection keeps its metric.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 5) Skip re-embedding if this collection already holds documents.
    #    collection.count() returns how many items are stored.
    existing = collection.count()
    if existing > 0:
        print(f"Collection '{COLLECTION_NAME}' already has {existing} documents "
              f"— skipping embedding (delete {DB_DIR} to rebuild).")
        return

    # 3) Load the embedding model. First run downloads it from Hugging Face
    #    (~90 MB) into a local cache; later runs load from disk.
    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # Pull the parallel lists we need. Order is preserved across all of them.
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {"source": c["source"], "type": c["type"], "chunk_id": c["chunk_id"]}
        for c in chunks
    ]

    # Embed EVERY chunk in one batched call. encode() internally batches and is
    # far faster than looping per chunk. convert_to_numpy gives an ndarray we
    # then .tolist() because Chroma wants plain Python lists.
    print(f"Embedding {len(documents)} chunks ...")
    start = time.time()
    embeddings = model.encode(
        documents,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()

    # Store everything. We use add() (not upsert()) because the count() guard
    # above guarantees the collection is empty here — add() raises on duplicate
    # ids, which is the safe behavior we want. (upsert() would silently
    # overwrite; use it only when intentionally updating existing ids.)
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    elapsed = time.time() - start

    # 6) Report.
    print(f"\nEmbedded and stored {len(ids)} chunks "
          f"in collection '{COLLECTION_NAME}'.")
    print(f"Time taken: {elapsed:.2f}s "
          f"({len(ids) / elapsed:.1f} chunks/sec)")


if __name__ == "__main__":
    # Pass reset=True when invoked as: python3 embed.py --reset
    main(reset="--reset" in sys.argv)

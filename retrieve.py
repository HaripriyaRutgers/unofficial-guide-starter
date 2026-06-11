"""
retrieve.py — Stage 4 of the RAG pipeline: Retrieval.

Embeds a query with the SAME model used at index time (all-MiniLM-L6-v2) and
runs semantic (nearest-neighbor) search against the persistent ChromaDB
"cs_careers" collection, returning the top-k chunks.

Run with:  python retrieve.py     (after running embed.py)
Libraries:  sentence-transformers, chromadb, json, time
"""

import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "./chroma_db"
COLLECTION_NAME = "cs_careers"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load the model and open the collection ONCE at import, not per query — the
# model load (~1-2s) and client connect are expensive, so we reuse them.
# IMPORTANT: the query must be embedded with the same model that built the
# index, or distances are meaningless.
_model = SentenceTransformer(MODEL_NAME)
_client = chromadb.PersistentClient(path=DB_DIR)
# get_collection (not get_or_create) so we fail loudly if embed.py hasn't run.
_collection = _client.get_collection(name=COLLECTION_NAME)


def retrieve(query: str, k: int = 5):
    """Return the top-k chunks most similar to `query`.

    Each result is a dict: {chunk_id, source, text, distance}.
    `distance` is COSINE distance (set in embed.py): 0 = identical direction,
    ~1 = unrelated, 2 = opposite. Lower is better.
    """
    # Embed the query with the same model used for the chunks.
    query_embedding = _model.encode([query], convert_to_numpy=True).tolist()

    # query() does the nearest-neighbor search. n_results = k.
    # Results come back as lists-of-lists (one inner list per query embedding);
    # we passed a single query, so we read index [0] of each field below.
    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    out = []
    # Zip the parallel result lists for our single query (the [0] slice).
    for chunk_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        out.append({
            "chunk_id": chunk_id,
            # Prefer metadata["source"]; fall back to the id if ever missing.
            "source": meta.get("source", "?"),
            "text": doc,
            "distance": dist,
        })
    return out


if __name__ == "__main__":
    test_queries = [
        "What does a UX researcher actually do day to day?",
        "What VFX roles are available to someone with a programming background?",
        "What non-SWE careers do CS students on forums talk about switching to?",
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {q}")
        print("=" * 80)
        for rank, r in enumerate(retrieve(q, k=5), start=1):
            print(f"RANK {rank} | source: {r['source']} | "
                  f"distance: {r['distance']:.4f} | chunk_id: {r['chunk_id']}")
            # First 200 chars of the chunk, newlines flattened for readability.
            preview = " ".join(r["text"].split())[:200]
            print(f"  {preview}")

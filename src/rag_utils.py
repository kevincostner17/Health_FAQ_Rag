"""Shared retrieval utilities for the Health FAQ RAG pipeline."""
from pathlib import Path

import chromadb

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = REPO_ROOT / "chroma_db"
COLLECTION_NAME = "medquad_cancergov"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 6
# Cosine distance: 0 = identical, 1 = unrelated. Chunks above this are dropped.
MAX_DISTANCE = 0.55

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def enrich_for_embedding(focus: str, question: str, chunk_text: str) -> str:
    """Embed focus + canonical question + chunk so queries match MedQuAD phrasing."""
    return f"Topic: {focus}. Question: {question}. Answer: {chunk_text}"


def encode_texts(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 100)
    return vectors.tolist()


def encode_query(question: str) -> list[float]:
    return encode_texts([question])[0]


def get_client():
    return chromadb.PersistentClient(path=str(DB_DIR))


def load_collection():
    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {DB_DIR}. Run: python src/03_embed_store.py"
        )
    client = get_client()
    return client.get_collection(name=COLLECTION_NAME)


def retrieve(question: str, collection, top_k: int = TOP_K, max_distance: float = MAX_DISTANCE):
    query_embedding = encode_query(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    retrieved = []
    for doc, meta, dist in zip(docs, metas, dists):
        if dist <= max_distance:
            retrieved.append((doc, meta, dist))
    return retrieved


def format_context(retrieved) -> str:
    if not retrieved:
        return ""
    blocks = []
    for doc, meta, _dist in retrieved:
        blocks.append(f"[Source: {meta['focus']} | {meta['source']}]\n{doc}")
    return "\n\n---\n\n".join(blocks)


def is_likely_out_of_kb(retrieved) -> bool:
    return len(retrieved) == 0

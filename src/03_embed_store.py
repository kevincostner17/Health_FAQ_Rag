"""
Step 3: Embed & Store
Embed each chunk with sentence-transformers and store in Chroma.

Run: python src/03_embed_store.py
Requires: data/chunks.json (produced by 02_chunk.py)
"""
import json
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_utils import (
    COLLECTION_NAME,
    DB_DIR,
    EMBEDDING_MODEL,
    enrich_for_embedding,
    encode_texts,
    get_embedder,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = REPO_ROOT / "data" / "chunks.json"
BATCH = 256


def main():
    if not CHUNKS_FILE.exists():
        print(f"ERROR: {CHUNKS_FILE} not found. Run 02_chunk.py first.")
        return

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["chunk_text"] for c in chunks]
    metadatas = [
        {
            "doc_id": c["doc_id"],
            "focus": c["focus"],
            "source": c["source"],
            "url": c["url"],
            "question": c["question"],
            "qtype": c["qtype"] or "",
        }
        for c in chunks
    ]
    embed_inputs = [
        enrich_for_embedding(c["focus"], c["question"], c["chunk_text"])
        for c in chunks
    ]

    print(f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL} ...")
    get_embedder()

    print(f"Initializing Chroma persistent client at {DB_DIR} ...")
    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )

    print(f"Storing chunks in batches of {BATCH} ...")
    for i in range(0, len(chunks), BATCH):
        batch_embs = encode_texts(embed_inputs[i : i + BATCH])
        collection.add(
            ids=ids[i : i + BATCH],
            documents=documents[i : i + BATCH],
            metadatas=metadatas[i : i + BATCH],
            embeddings=batch_embs,
        )
        print(f"  stored {min(i + BATCH, len(chunks))}/{len(chunks)}")

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")


if __name__ == "__main__":
    main()

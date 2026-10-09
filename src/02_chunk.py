"""
Step 2: Chunking
Split answer passages into retrieval-sized chunks with overlap, using LangChain's
RecursiveCharacterTextSplitter (splits on paragraph -> sentence -> word boundaries).

Run: python 02_chunk.py
Requires: data/qa_pairs.json (produced by 01_ingest.py)
"""
import json
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = REPO_ROOT / "data" / "qa_pairs.json"
OUTPUT_FILE = REPO_ROOT / "data" / "chunks.json"

CHUNK_SIZE = 500       # characters (~roughly 100-125 tokens)
CHUNK_OVERLAP = 75     # characters


def clean_text(text: str) -> str:
    """Collapse MedQuAD's messy whitespace/newlines into clean text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run 01_ingest.py first.")
        return

    with open(INPUT_FILE, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    seen_ids = set()
    for idx, record in enumerate(qa_pairs):
        clean_answer = clean_text(record["answer"])
        pieces = splitter.split_text(clean_answer)

        for i, piece in enumerate(pieces):
            chunk_id = f"rec{idx}_{record['qid']}_c{i}"
            suffix = 0
            unique_id = chunk_id
            while unique_id in seen_ids:
                suffix += 1
                unique_id = f"{chunk_id}_{suffix}"
            seen_ids.add(unique_id)

            all_chunks.append({
                "chunk_id": unique_id,
                "doc_id": record["doc_id"],
                "focus": record["focus"],
                "source": record["source"],
                "url": record["url"],
                "question": record["question"],
                "qtype": record["qtype"],
                "chunk_text": piece,
                "chunk_index": i,
                "total_chunks": len(pieces),
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Input:  {len(qa_pairs)} Q-A pairs")
    print(f"Output: {len(all_chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"Avg chunks per answer: {len(all_chunks)/len(qa_pairs):.2f}")
    print(f"Saved to {OUTPUT_FILE}")

    print("\n--- Sample chunk ---")
    print(json.dumps(all_chunks[0], indent=2)[:600])


if __name__ == "__main__":
    main()

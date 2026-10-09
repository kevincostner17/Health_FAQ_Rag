"""
Step 4: Retrieval sanity check.
Run a handful of sample questions through the vector store and inspect
whether the top retrieved chunks are actually relevant.

Run: python src/04_retrieve_test.py
Requires: 03_embed_store.py to have been run first (creates chroma_db/)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_utils import MAX_DISTANCE, load_collection, retrieve

TEST_QUESTIONS = [
    "What are the symptoms of leukemia?",              # Known — in KB
    "How is breast cancer treated?",                    # Known — in KB
    "What causes lung cancer?",                          # Known — in KB
    "What is the capital of France?",                    # Out-of-KB
    "What's the best diet for training a marathon?",     # Out-of-KB
]


def main():
    try:
        collection = load_collection()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    for question in TEST_QUESTIONS:
        print("=" * 80)
        print(f"Q: {question}")
        retrieved = retrieve(question, collection)

        if not retrieved:
            print(f"  -> No relevant chunks within distance {MAX_DISTANCE} (likely Out-of-KB)")
        else:
            for doc, meta, dist in retrieved[:3]:
                print(f"\n  [distance={dist:.3f}] focus='{meta['focus']}'")
                print(f"  qtype='{meta['qtype']}' | question='{meta['question'][:70]}...'")
                print(f"  chunk: {doc[:180]}...")
        print()


if __name__ == "__main__":
    main()

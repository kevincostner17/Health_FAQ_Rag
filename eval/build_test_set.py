"""
Build Walert-style evaluation test sets from qa_pairs.json.

Run: python eval/build_test_set.py
Output: data/test_sets.json
"""
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QA_FILE = REPO_ROOT / "data" / "qa_pairs.json"
OUT_FILE = REPO_ROOT / "data" / "test_sets.json"

KNOWN_PER_QTYPE = 4
OUT_OF_KB = [
    {"question": "What is the capital of France?"},
    {"question": "What's the best diet for training a marathon?"},
    {"question": "How do I fix a flat tire on my bicycle?"},
    {"question": "Who won the FIFA World Cup in 2022?"},
    {"question": "How do I learn to play the guitar?"},
]

INFERRED = [
    {
        "question": "What symptoms and risk factors are associated with lung cancer?",
        "note": "Requires combining symptoms + susceptibility chunks",
    },
    {
        "question": "How is breast cancer diagnosed and what treatments are available?",
        "note": "Requires combining exams/tests + treatment chunks",
    },
    {
        "question": "What are the signs of leukemia and how is it diagnosed?",
        "note": "Requires combining symptoms + diagnostic test chunks",
    },
]


def main():
    with open(QA_FILE, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    by_qtype: dict[str, list] = {}
    for rec in qa_pairs:
        qtype = rec.get("qtype") or "general"
        by_qtype.setdefault(qtype, []).append(rec)

    rng = random.Random(42)
    known = []
    for qtype, records in sorted(by_qtype.items()):
        sample = records if len(records) <= KNOWN_PER_QTYPE else rng.sample(records, KNOWN_PER_QTYPE)
        for rec in sample:
            known.append({
                "question": rec["question"],
                "gold_answer": rec["answer"],
                "focus": rec["focus"],
                "qtype": qtype,
                "url": rec["url"],
            })

    payload = {
        "known": known,
        "inferred": INFERRED,
        "out_of_kb": OUT_OF_KB,
        "stats": {
            "known_count": len(known),
            "inferred_count": len(INFERRED),
            "out_of_kb_count": len(OUT_OF_KB),
            "source_qa_pairs": len(qa_pairs),
        },
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Built test set -> {OUT_FILE}")
    print(f"  Known:     {len(known)}")
    print(f"  Inferred:  {len(INFERRED)}")
    print(f"  Out-of-KB: {len(OUT_OF_KB)}")


if __name__ == "__main__":
    main()

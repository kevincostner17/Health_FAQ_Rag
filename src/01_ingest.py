"""
Step 1: Ingestion
Parse MedQuAD XML files (CancerGov subset) into clean question-answer-source records.

Run: python 01_ingest.py
Requires: the MedQuAD/1_CancerGov_QA folder to be present in this same directory
          (download from https://github.com/abachaa/MedQuAD if not already present).
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "MedQuAD" / "1_CancerGov_QA"
OUTPUT_FILE = REPO_ROOT / "data" / "qa_pairs.json"


def parse_medquad_file(filepath: Path):
    """Extract Q-A pairs from a single MedQuAD XML document."""
    records = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  [skip] {filepath.name}: {e}")
        return records

    doc_id = root.get("id")
    source = root.get("source")
    url = root.get("url")
    focus_elem = root.find("Focus")
    focus = focus_elem.text.strip() if focus_elem is not None and focus_elem.text else "Unknown"

    for qa_pair in root.findall(".//QAPair"):
        q_elem = qa_pair.find("Question")
        a_elem = qa_pair.find("Answer")
        if q_elem is None or a_elem is None:
            continue
        question = (q_elem.text or "").strip()
        answer = (a_elem.text or "").strip()
        if not question or not answer:
            continue
        records.append({
            "doc_id": doc_id,
            "qid": q_elem.get("qid"),
            "qtype": q_elem.get("qtype"),
            "focus": focus,
            "source": source,
            "url": url,
            "question": question,
            "answer": answer,
        })
    return records


def main():
    if not SOURCE_DIR.exists():
        print(f"ERROR: {SOURCE_DIR} not found.")
        print("Download it first:")
        print(f"  git clone https://github.com/abachaa/MedQuAD.git \"{REPO_ROOT / 'MedQuAD'}\"")
        return

    files = sorted(SOURCE_DIR.glob("*.xml"))
    print(f"Found {len(files)} XML files in {SOURCE_DIR}")

    all_records = []
    for f in files:
        all_records.extend(parse_medquad_file(f))

    print(f"Parsed {len(all_records)} question-answer pairs")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    print(f"Saved to {OUTPUT_FILE}")

    if all_records:
        print("\n--- Sample record ---")
        sample = all_records[0]
        print(f"Focus: {sample['focus']}")
        print(f"Question: {sample['question']}")
        print(f"Answer (first 200 chars): {sample['answer'][:200]}...")


if __name__ == "__main__":
    main()

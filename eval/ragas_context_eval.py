"""
RAGAS context metrics (context precision + context recall) on a sample of known questions,
reported alongside faithfulness / relevance and the out-of-scope detector's sensitivity.

Reuses the answers and RAGAS scores saved by the full evaluation (data/eval_results.json),
so only the two context metrics need new judge calls (fits the Groq free-tier daily limit).

Run: python eval/ragas_context_eval.py   (after python src/run_eval.py)
"""
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

import os

from backend.ragas_metrics import JUDGE_MODEL, _get_judge_and_embeddings
from rag_utils import load_collection, retrieve

MAX_PER_TYPE = 3      # sample: at most 3 questions per question type, in test-set order
SAMPLE_SIZE = 15


def _select_sample(eval_results, test_sets):
    """Known questions that already have real RAGAS faithfulness/relevance scores."""
    gold = {x["question"]: x for x in test_sets["known"]}
    per_type, sample = Counter(), []
    for item in eval_results["known"]:
        ragas = item.get("ragas") or {}
        if ragas.get("method") != "ragas":
            continue
        qtype = gold[item["question"]].get("qtype")
        if per_type[qtype] >= MAX_PER_TYPE:
            continue
        per_type[qtype] += 1
        sample.append({**item, "qtype": qtype, "reference": gold[item["question"]]["gold_answer"]})
        if len(sample) == SAMPLE_SIZE:
            break
    return sample


def _avg(values):
    return round(sum(values) / len(values), 3) if values else None


def main():
    eval_path = REPO_ROOT / "data" / "eval_results.json"
    if not eval_path.exists():
        print("Run the full evaluation first: python src/run_eval.py")
        sys.exit(1)
    eval_results = json.load(open(eval_path, encoding="utf-8"))
    test_sets = json.load(open(REPO_ROOT / "data" / "test_sets.json", encoding="utf-8"))
    sample = _select_sample(eval_results, test_sets)

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall
    from ragas.run_config import RunConfig

    llm, embeddings = _get_judge_and_embeddings(os.environ["GROQ_API_KEY"])
    collection = load_collection()

    print("=" * 70)
    print(f"RAGAS context metrics on {len(sample)} known questions (judge: {JUDGE_MODEL})")
    print("=" * 70)

    rows = []
    for item in sample:
        q = item["question"]
        contexts = [doc for doc, _meta, _dist in retrieve(q, collection)]
        try:
            result = evaluate(
                Dataset.from_dict({"question": [q], "contexts": [contexts], "ground_truth": [item["reference"]],
                                   "answer": [item["answer"]]}),
                metrics=[context_precision, context_recall],
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=True,
                show_progress=False,
                run_config=RunConfig(timeout=900, max_retries=10, max_wait=60, max_workers=1),
            )
            df = result.to_pandas()
            cp, cr = float(df["context_precision"].iloc[0]), float(df["context_recall"].iloc[0])
            error = None
        except Exception as e:
            cp = cr = None
            error = f"{type(e).__name__}: {str(e)[:200]}"
        row = {"question": q, "qtype": item["qtype"], "n_contexts": len(contexts),
               "faithfulness": item["ragas"]["faithfulness"], "relevance": item["ragas"]["relevance"],
               "context_precision": None if cp is None else round(cp, 3),
               "context_recall": None if cr is None else round(cr, 3), "error": error}
        rows.append(row)
        print(f"\n[{item['qtype']}] {q}")
        print(f"  faith={row['faithfulness']} rel={row['relevance']} "
              f"ctx_precision={row['context_precision']} ctx_recall={row['context_recall']}"
              + (f"  ERROR {error}" if error else ""))

    ok = [r for r in rows if r["error"] is None]
    s = eval_results["summary"]
    oos_refused, oos_total = map(int, s["out_of_kb_refused"].split("/"))
    known_answered, known_total = map(int, s["known_answered"].split("/"))
    summary = {
        "judge_model": JUDGE_MODEL,
        "sample_size": len(rows),
        "scored": len(ok),
        "avg_faithfulness": _avg([r["faithfulness"] for r in ok]),
        "avg_relevance": _avg([r["relevance"] for r in ok]),
        "avg_context_precision": _avg([r["context_precision"] for r in ok]),
        "avg_context_recall": _avg([r["context_recall"] for r in ok]),
        # Out-of-scope detector treated as a classifier: positive = "question is out of scope".
        "sensitivity": round(oos_refused / oos_total, 3),            # out-of-scope questions refused
        "specificity": round(known_answered / known_total, 3),       # in-scope questions answered
        "sensitivity_counts": s["out_of_kb_refused"],
        "specificity_counts": s["known_answered"],
    }
    out = REPO_ROOT / "data" / "ragas_context_results.json"
    json.dump({"summary": summary, "results": rows}, open(out, "w", encoding="utf-8"), indent=2)

    print("\n" + "=" * 70)
    for k in ("scored", "avg_faithfulness", "avg_relevance", "avg_context_precision", "avg_context_recall",
              "sensitivity", "specificity"):
        print(f"{k:24s} {summary[k]}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

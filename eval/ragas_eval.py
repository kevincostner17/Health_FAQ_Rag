"""
Batch RAGAS evaluation on test sets.

Run: python eval/ragas_eval.py [--quick]
"""
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from backend.ragas_metrics import JUDGE_MODEL, compute_metrics
from rag_utils import load_collection


def _load_generate():
    spec = importlib.util.spec_from_file_location("generate", REPO_ROOT / "src" / "05_generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sources(retrieved):
    return [
        {
            "focus": m["focus"],
            "source": m["source"],
            "url": m["url"],
            "question": m["question"],
            "chunk": d,
            "distance": round(float(dist), 3),
        }
        for d, m, dist in retrieved
    ]


def main():
    quick = "--quick" in sys.argv
    test_path = REPO_ROOT / "data" / "test_sets.json"
    if not test_path.exists():
        print("Run: python eval/build_test_set.py first")
        sys.exit(1)

    with open(test_path, encoding="utf-8") as f:
        sets = json.load(f)

    samples = sets["known"][:5 if quick else 15]
    gen = _load_generate()
    collection = load_collection()

    from rag_utils import format_context, is_likely_out_of_kb, retrieve

    results = []
    faith_scores, rel_scores, hall_scores = [], [], []

    print("=" * 70)
    print(f"RAGAS batch eval ({len(samples)} known questions)")
    print("=" * 70)

    for item in samples:
        q = item["question"]
        retrieved = retrieve(q, collection)
        refused = is_likely_out_of_kb(retrieved)
        sources = _sources(retrieved)
        if refused:
            answer = gen.OUT_OF_KB_REPLY
        else:
            answer = gen.generate_answer(q, format_context(retrieved))

        metrics = compute_metrics(q, answer, sources, refused, patient=True)
        results.append({"question": q, "metrics": metrics, "refused": refused})

        # Only real RAGAS scores count; heuristic fallbacks are listed but not averaged.
        if metrics.get("method") == "ragas":
            faith_scores.append(metrics["faithfulness"])
            rel_scores.append(metrics["relevance"])
            hall_scores.append(metrics["hallucination"])

        print(f"\nQ: {q[:70]}")
        print(f"   Faithfulness: {metrics['faithfulness']}  Relevance: {metrics['relevance']}  Hallucination: {metrics['hallucination']}  ({metrics['method']})")
        if metrics.get("error"):
            print(f"   RAGAS error: {metrics['error']}")

    summary = {
        "count": len(results),
        "judge_model": JUDGE_MODEL,
        "ragas_scored": len(faith_scores),
        "refused": sum(1 for r in results if r["refused"]),
        "model_refusal": sum(1 for r in results if r["metrics"].get("method") == "model_refusal"),
        "heuristic_fallback": sum(1 for r in results if r["metrics"].get("method") == "heuristic"),
        "avg_faithfulness": round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else 0,
        "avg_relevance": round(sum(rel_scores) / len(rel_scores), 3) if rel_scores else 0,
        "avg_hallucination": round(sum(hall_scores) / len(hall_scores), 3) if hall_scores else 0,
        "results": results,
    }

    out = REPO_ROOT / "data" / "ragas_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"RAGAS scored:      {summary['ragas_scored']}/{summary['count']} (refused: {summary['refused']}, model declined: {summary['model_refusal']}, heuristic fallback: {summary['heuristic_fallback']})")
    print(f"Avg Faithfulness:  {summary['avg_faithfulness']}")
    print(f"Avg Relevance:     {summary['avg_relevance']}")
    print(f"Avg Hallucination: {summary['avg_hallucination']}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

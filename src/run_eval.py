"""
Full Walert-style evaluation: Known (ROUGE-1/BERTScore), Inferred, Out-of-KB (refusal %).

Run:
  python eval/build_test_set.py   # once, to create data/test_sets.json
  python src/run_eval.py
"""
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from metrics import bertscore_f1, looks_like_refusal, rouge1_score
from rag_utils import format_context, is_likely_out_of_kb, load_collection, retrieve

try:
    from backend.ragas_metrics import compute_metrics
except ImportError:
    compute_metrics = None


def _load_generate_module():
    path = Path(__file__).parent / "05_generate.py"
    spec = importlib.util.spec_from_file_location("generate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_test_sets():
    path = REPO_ROOT / "data" / "test_sets.json"
    if not path.exists():
        print("Test sets not found. Run: python eval/build_test_set.py")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _truncate(text: str, n: int = 300) -> str:
    return text[:n] + ("..." if len(text) > n else "")


def main():
    quick = "--quick" in sys.argv
    gen = _load_generate_module()
    sets = _load_test_sets()
    if quick:
        sets = {
            **sets,
            "known": sets["known"][:10],
            "inferred": sets["inferred"][:2],
            "out_of_kb": sets["out_of_kb"][:3],
        }
        print("Quick eval mode (subset of test questions)\n")

    try:
        collection = load_collection()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    results = {"known": [], "inferred": [], "out_of_kb": [], "summary": {}}
    ragas_faith, ragas_rel, ragas_hall = [], [], []
    ragas_attempted = 0

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

    # --- Known ---
    print("=" * 80)
    print(f"KNOWN QUESTIONS ({len(sets['known'])}) — ROUGE-1 / BERTScore vs gold answer")
    print("=" * 80)
    known_ok = 0
    rouge_scores = []
    bert_scores = []

    for item in sets["known"]:
        q = item["question"]
        gold = item["gold_answer"]
        try:
            answer = gen.answer_question(q, collection)
            refused = looks_like_refusal(answer)
            rouge = bert = None
            if not refused and gold:
                rouge = rouge1_score(answer, gold)
                rouge_scores.append(rouge)
                try:
                    bert = bertscore_f1(answer, gold)
                    bert_scores.append(bert)
                except Exception as e:
                    bert = f"error: {e}"

            status = "OK" if not refused else "REFUSED"
            if not refused:
                known_ok += 1
            print(f"\n[{status}] ({item.get('qtype')}) {q}")
            if rouge is not None:
                print(f"  ROUGE-1: {rouge:.3f}  BERTScore F1: {bert:.3f}" if isinstance(bert, float) else f"  ROUGE-1: {rouge:.3f}  BERTScore: {bert}")
            print(f"  A: {_truncate(answer)}")
            ragas = None
            if compute_metrics and not refused:
                retrieved = retrieve(q, collection)
                ragas = compute_metrics(q, answer, _sources(retrieved), refused, patient=True)
                ragas_attempted += 1
                # Only real RAGAS scores go into the averages; heuristic fallbacks are reported separately.
                if ragas["method"] == "ragas":
                    ragas_faith.append(ragas["faithfulness"])
                    ragas_rel.append(ragas["relevance"])
                    ragas_hall.append(ragas["hallucination"])
                print(f"  RAGAS faith={ragas['faithfulness']} rel={ragas['relevance']} hall={ragas['hallucination']} ({ragas['method']})")
                if ragas.get("error"):
                    print(f"  RAGAS error: {ragas['error']}")

            results["known"].append({
                "question": q, "refused": refused, "rouge1": rouge,
                "bertscore_f1": bert, "answer": answer, "focus": item.get("focus"),
                "ragas": ragas,
            })
        except RuntimeError as e:
            print(f"\n[ERROR] {q}\n  {e}")
            results["known"].append({"question": q, "error": str(e)})

    # --- Inferred ---
    print("\n" + "=" * 80)
    print(f"INFERRED QUESTIONS ({len(sets['inferred'])}) — synthesis across chunks")
    print("=" * 80)
    inferred_ok = 0
    for item in sets["inferred"]:
        q = item["question"]
        try:
            answer = gen.answer_question(q, collection)
            refused = looks_like_refusal(answer)
            status = "OK" if not refused else "REFUSED"
            if not refused:
                inferred_ok += 1
            print(f"\n[{status}] {q}")
            print(f"  A: {_truncate(answer)}")
            results["inferred"].append({"question": q, "refused": refused, "answer": answer})
        except RuntimeError as e:
            print(f"\n[ERROR] {q}\n  {e}")

    # --- Out-of-KB ---
    print("\n" + "=" * 80)
    print(f"OUT-OF-KB QUESTIONS ({len(sets['out_of_kb'])}) — should refuse")
    print("=" * 80)
    refused_count = 0
    for item in sets["out_of_kb"]:
        q = item["question"]
        try:
            answer = gen.answer_question(q, collection)
            refused = looks_like_refusal(answer)
            status = "OK (refused)" if refused else "WARN (answered)"
            if refused:
                refused_count += 1
            print(f"\n[{status}] {q}")
            print(f"  A: {_truncate(answer)}")
            results["out_of_kb"].append({"question": q, "refused": refused, "answer": answer})
        except RuntimeError as e:
            print(f"\n[ERROR] {q}\n  {e}")

    # --- Summary ---
    n_known = len(sets["known"])
    n_inferred = len(sets["inferred"])
    n_oob = len(sets["out_of_kb"])
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    avg_bert = sum(bert_scores) / len(bert_scores) if bert_scores else 0
    oob_pct = refused_count / n_oob * 100 if n_oob else 0

    results["summary"] = {
        "known_answered": f"{known_ok}/{n_known}",
        "known_answer_rate_pct": round(known_ok / n_known * 100, 1) if n_known else 0,
        "avg_rouge1": round(avg_rouge, 3),
        "avg_bertscore_f1": round(avg_bert, 3),
        "avg_ragas_faithfulness": round(sum(ragas_faith) / len(ragas_faith), 3) if ragas_faith else None,
        "avg_ragas_relevance": round(sum(ragas_rel) / len(ragas_rel), 3) if ragas_rel else None,
        "avg_ragas_hallucination": round(sum(ragas_hall) / len(ragas_hall), 3) if ragas_hall else None,
        "ragas_scored": f"{len(ragas_faith)}/{ragas_attempted}",
        "inferred_answered": f"{inferred_ok}/{n_inferred}",
        "out_of_kb_refusal_rate_pct": round(oob_pct, 1),
        "out_of_kb_refused": f"{refused_count}/{n_oob}",
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Known answered:        {known_ok}/{n_known} ({results['summary']['known_answer_rate_pct']}%)")
    print(f"Avg ROUGE-1:           {avg_rouge:.3f}")
    print(f"Avg BERTScore F1:      {avg_bert:.3f}")
    if ragas_attempted:
        print(f"RAGAS scored:          {len(ragas_faith)}/{ragas_attempted} (rest fell back to heuristic)")
    if ragas_faith:
        print(f"Avg RAGAS Faithfulness: {results['summary']['avg_ragas_faithfulness']}")
        print(f"Avg RAGAS Relevance:    {results['summary']['avg_ragas_relevance']}")
        print(f"Avg RAGAS Hallucination:{results['summary']['avg_ragas_hallucination']}")
    print(f"Inferred answered:     {inferred_ok}/{n_inferred}")
    print(f"Out-of-KB refusal:     {refused_count}/{n_oob} ({oob_pct:.0f}%)")
    print("=" * 80)

    out_file = REPO_ROOT / "data" / "eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()

"""
Retrieval-only comparison of reranking set-ups for V4 (no LLM calls).

Configurations are chosen on the development set (data/dev_questions.json) and only then
checked on the test set (data/eval_questions.json), so the reported test numbers are not tuned.

  none        Chroma top 5, no reranking (= V3)
  ce          cross-encoder on 20 candidates, keep 5 (original V4)
  ce_k10      cross-encoder on 10 candidates
  ce_text     cross-encoder sees only the chunk text, without the "Topic/Question" prefix
  rrf         reciprocal rank fusion of the vector rank and the cross-encoder rank (20 candidates)
  rrf_k10     reciprocal rank fusion on 10 candidates

Run: python eval/rerank_ablation.py
Output: data/rerank_ablation.json
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from compare_versions import retrieval_metrics
from versions.common import FINAL_K, ScoredRetriever
from versions.v3_chroma import get_vectorstore
from versions.v4_rerank import get_cross_encoder

RRF_K = 60
OUT_FILE = REPO_ROOT / "data" / "rerank_ablation.json"


def ce_scores(question, docs, text_only=False):
    if not docs:
        return []
    texts = [d.metadata["chunk_text"] if text_only else d.page_content for d in docs]
    return list(get_cross_encoder().predict([(question, t) for t in texts]))


def run_config(name, question, cands20):
    if name == "none":
        return cands20[:FINAL_K]
    k = 10 if name.endswith("k10") else 20
    docs = cands20[:k]
    scores = ce_scores(question, docs, text_only=(name == "ce_text"))
    if name.startswith("ce"):
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    else:  # rrf: vector rank is the list order; cross-encoder rank from scores
        ce_rank = {i: r for r, i in enumerate(sorted(range(len(docs)), key=lambda i: scores[i], reverse=True))}
        fused = {i: 1 / (RRF_K + i + 1) + 1 / (RRF_K + ce_rank[i] + 1) for i in range(len(docs))}
        order = sorted(fused, key=fused.get, reverse=True)
    return [docs[i] for i in order[:FINAL_K]]


def evaluate(items, configs, retriever):
    totals = {c: {"hit": 0.0, "rr": 0.0, "precision": 0.0} for c in configs}
    for item in items:
        cands = retriever.invoke(item["question"])
        for c in configs:
            sources = [{"question": d.metadata["question"]} for d in run_config(c, item["question"], cands)]
            m = retrieval_metrics(sources, item["gold_questions"])
            for key in totals[c]:
                totals[c][key] += float(m[key])
    n = len(items)
    return {c: {k: round(v / n, 3) for k, v in t.items()} for c, t in totals.items()}


def main():
    configs = ["none", "ce", "ce_k10", "ce_text", "rrf", "rrf_k10"]
    retriever = ScoredRetriever(vectorstore=get_vectorstore(), k=20)
    results = {}
    for name, path in (("dev", "dev_questions.json"), ("test", "eval_questions.json")):
        items = json.loads((REPO_ROOT / "data" / path).read_text())["in_scope"]
        results[name] = {"n": len(items), "configs": evaluate(items, configs, retriever)}

    print(f"{'config':10} | {'dev hit@5':>9} {'dev MRR':>8} {'dev P@5':>8} | {'test hit@5':>10} {'test MRR':>9} {'test P@5':>9}")
    for c in configs:
        d, t = results["dev"]["configs"][c], results["test"]["configs"][c]
        print(f"{c:10} | {d['hit']:9.3f} {d['rr']:8.3f} {d['precision']:8.3f} | "
              f"{t['hit']:10.3f} {t['rr']:9.3f} {t['precision']:9.3f}")
    print(f"(dev n={results['dev']['n']}, test n={results['test']['n']})")
    OUT_FILE.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

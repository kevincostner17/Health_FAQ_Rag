"""
Compare V2 (FAISS), V3 (ChromaDB) and V4 (reranking) on the same questions.
V1 (no retrieval) can be added as a baseline with --versions v1,v2,v3,v4: it has no retrieval
metrics, but shows what grounding adds to accuracy, sources and refusals.

Measures, per version:
  Retrieval quality  hit@5, MRR, precision@5 (chunk is relevant if it comes from a gold answer)
  Answer accuracy    LLM judge grade (correct 1 / partial 0.5 / wrong 0) and BERTScore F1
  Refusals           out-of-scope refusal rate, in-scope false refusals
  Response time      retrieve / rerank / generate / total, mean and median

Questions are interleaved across versions so network drift affects all versions equally.
Every result is cached in data/version_comparison_raw.json, so a run stopped by Groq rate
limits resumes where it left off. Judge scores are attached to LangSmith traces as feedback
when LangSmith tracing is enabled.

Run: python eval/compare_versions.py [--limit N] [--no-judge] [--fresh] [--pace SECONDS]
Output: data/version_comparison.json, docs/version_comparison.md
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from metrics import looks_like_refusal
from versions import v1_basics, v2_faiss, v3_chroma, v4_rerank

QUESTIONS_FILE = REPO_ROOT / "data" / "eval_questions.json"
QA_FILE = REPO_ROOT / "data" / "qa_pairs.json"
RAW_FILE = REPO_ROOT / "data" / "version_comparison_raw.json"
OUT_JSON = REPO_ROOT / "data" / "version_comparison.json"
OUT_MD = REPO_ROOT / "docs" / "version_comparison.md"



class _V1Adapter:
    """Gives V1's structured answer the same result shape as V2-V4 (no sources)."""

    @staticmethod
    def ask(question: str, config: dict | None = None) -> dict:
        start = time.perf_counter()
        result = v1_basics.ask(question, config=config)
        elapsed = round((time.perf_counter() - start) * 1000, 1)
        answer = result.answer + "\n" + "\n".join(f"- {p}" for p in result.key_points)
        return {"answer": answer, "refused": False, "sources": [],
                "timings": {"generate_ms": elapsed, "total_ms": elapsed}}


VERSIONS = {"v1": _V1Adapter, "v2": v2_faiss, "v3": v3_chroma, "v4": v4_rerank}
LABELS = {"v1": "V1 No retrieval", "v2": "V2 FAISS", "v3": "V3 ChromaDB", "v4": "V4 Reranking"}
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-20b")
GRADE_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}

JUDGE_SYSTEM = """You grade answers from a cancer-information chatbot against reference answers \
from the US National Cancer Institute.
Grades:
- "correct": answers the question and its main points agree with the reference.
- "partial": on topic but misses key points, or answers about a related but different condition.
- "wrong": incorrect, off topic, contradicts the reference, or declines to answer.
Extra correct detail is fine. Reply with JSON only: {{"grade": "correct|partial|wrong", "reason": "<one sentence>"}}"""

JUDGE_HUMAN = """Question: {question}

Reference answer(s):
{reference}

Chatbot answer:
{answer}"""


# ------------------------------------------------------------------ retrieval metrics

def is_refusal(rec: dict) -> bool:
    return rec["refused"] or looks_like_refusal(rec["answer"]) or "only help with health" in rec["answer"].lower()


def retrieval_metrics(sources: list[dict], gold_questions: list[str]) -> dict:
    gold = set(gold_questions)
    relevant = [s["question"] in gold for s in sources]
    first = next((i + 1 for i, r in enumerate(relevant) if r), None)
    return {
        "hit": first is not None,
        "rank": first,
        "rr": 1 / first if first else 0.0,
        "precision": sum(relevant) / len(relevant) if relevant else 0.0,
    }


# ------------------------------------------------------------------ caching

def load_raw(fresh: bool) -> dict:
    if fresh or not RAW_FILE.exists():
        return {}
    return json.loads(RAW_FILE.read_text())


def save_raw(raw: dict) -> None:
    RAW_FILE.write_text(json.dumps(raw, indent=2))


def slim_sources(sources: list[dict]) -> list[dict]:
    keep = ("focus", "question", "doc_type", "page", "similarity", "rerank_score", "url")
    return [{k: s[k] for k in keep if k in s} for s in sources]


# ------------------------------------------------------------------ phase 1: answers

def generate_answers(items, versions, raw, pace):
    from langchain_core.tracers.context import collect_runs

    for item in items:
        for v in versions:
            key = f"{v}:{item['id']}"
            if key in raw and "error" not in raw[key]:
                continue
            print(f"[{key}] {item['question'][:70]}")
            try:
                with collect_runs() as runs:
                    result = VERSIONS[v].ask(
                        item["question"],
                        config={"tags": ["eval"], "metadata": {"question_id": item["id"]}},
                    )
                run_id = str(runs.traced_runs[0].id) if runs.traced_runs else None
                raw[key] = {
                    "answer": result["answer"],
                    "refused": result["refused"],
                    "sources": slim_sources(result["sources"]),
                    "timings": result["timings"],
                    "run_id": run_id,
                }
                print(f"    refused={result['refused']} timings={result['timings']}")
                called_llm = not result["refused"]
            except Exception as e:  # keep going; the cache lets a later run retry
                raw[key] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                print(f"    ERROR {raw[key]['error']}")
                called_llm = True
            save_raw(raw)
            if called_llm and pace:
                time.sleep(pace)


# ------------------------------------------------------------------ phase 2: BERTScore

def add_bertscore(items, versions, raw, gold_answers):
    pairs = []
    for item in items:
        for v in versions:
            rec = raw.get(f"{v}:{item['id']}")
            if not rec or "error" in rec or "bertscore_f1" in rec:
                continue
            for ref in gold_answers[item["id"]]:
                pairs.append((f"{v}:{item['id']}", rec["answer"], ref))
    if not pairs:
        return
    from bert_score import score as bert_score

    print(f"BERTScore on {len(pairs)} answer/reference pairs ...")
    _, _, f1 = bert_score([p[1] for p in pairs], [p[2] for p in pairs], lang="en", verbose=False)
    best: dict[str, float] = {}
    for (key, _, _), value in zip(pairs, f1.tolist()):
        best[key] = max(best.get(key, 0.0), value)
    for key, value in best.items():
        raw[key]["bertscore_f1"] = round(value, 4)
    save_raw(raw)


# ------------------------------------------------------------------ phase 3: LLM judge

def get_judge():
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq

    prompt = ChatPromptTemplate.from_messages([("system", JUDGE_SYSTEM), ("human", JUDGE_HUMAN)])
    llm = ChatGroq(model=JUDGE_MODEL, temperature=0, reasoning_effort="low",
                   max_tokens=1024, max_retries=8)
    return (prompt | llm | JsonOutputParser()).with_config(run_name="Eval judge", tags=["judge"])


def judge_answers(items, versions, raw, gold_answers, pace):
    judge = None
    for item in items:
        for v in versions:
            key = f"{v}:{item['id']}"
            rec = raw.get(key)
            if not rec or "error" in rec or rec.get("judge", {}).get("grade") in GRADE_SCORE:
                continue
            if rec["refused"]:
                rec["judge"] = {"grade": "wrong", "reason": "Refused an in-scope question."}
                save_raw(raw)
                continue
            judge = judge or get_judge()
            reference = "\n\n".join(a[:2500] for a in gold_answers[item["id"]][:2])
            try:
                verdict = judge.invoke({"question": item["question"], "reference": reference,
                                        "answer": rec["answer"]})
                grade = str(verdict.get("grade", "")).lower().strip()
                rec["judge"] = {"grade": grade if grade in GRADE_SCORE else None,
                                "reason": verdict.get("reason", "")}
            except Exception as e:
                rec["judge"] = {"grade": None, "reason": f"judge error: {type(e).__name__}: {e}"[:300]}
            print(f"[{key}] judge: {rec['judge']['grade']} - {rec['judge']['reason'][:90]}")
            save_raw(raw)
            if pace:
                time.sleep(pace)


def send_feedback(items, versions, raw):
    if os.environ.get("LANGSMITH_TRACING", "").lower() != "true" or not os.environ.get("LANGSMITH_API_KEY"):
        return
    from langsmith import Client

    client = Client()
    sent = 0
    for item in items:
        for v in versions:
            rec = raw.get(f"{v}:{item['id']}") or {}
            if not rec.get("run_id") or rec.get("feedback_sent"):
                continue
            grade = (rec.get("judge") or {}).get("grade")
            if grade in GRADE_SCORE:
                client.create_feedback(rec["run_id"], key="judge_accuracy",
                                       score=GRADE_SCORE[grade], comment=rec["judge"]["reason"])
            if rec.get("retrieval"):
                client.create_feedback(rec["run_id"], key="hit_at_5", score=float(rec["retrieval"]["hit"]))
                client.create_feedback(rec["run_id"], key="reciprocal_rank", score=rec["retrieval"]["rr"])
            if "correct_refusal" in rec:
                client.create_feedback(rec["run_id"], key="correct_refusal", score=float(rec["correct_refusal"]))
            rec["feedback_sent"] = True
            sent += 1
    save_raw(raw)
    print(f"LangSmith: feedback attached to {sent} traces.")


# ------------------------------------------------------------------ summary + report

def _mean(values):
    values = [x for x in values if x is not None]
    return round(statistics.mean(values), 3) if values else None


def _median(values):
    values = [x for x in values if x is not None]
    return round(statistics.median(values), 1) if values else None


def summarise(in_scope, out_scope, versions, raw):
    summary = {}
    for v in versions:
        recs = [raw.get(f"{v}:{q['id']}") for q in in_scope]
        recs = [r for r in recs if r and "error" not in r]
        oos = [raw.get(f"{v}:{q['id']}") for q in out_scope]
        oos = [r for r in oos if r and "error" not in r]
        everything = recs + oos
        grades = [r.get("judge", {}).get("grade") for r in recs]

        def timing(name, pool=everything):
            vals = [r["timings"].get(name) for r in pool if r["timings"].get(name) is not None]
            return {"mean": round(statistics.mean(vals), 1) if vals else None, "median": _median(vals)}

        answered = [r for r in recs if not r["refused"]]
        summary[v] = {
            "label": LABELS[v],
            "in_scope_scored": len(recs),
            "hit_at_5": _mean([float(r["retrieval"]["hit"]) for r in recs if r.get("retrieval")]),
            "mrr": _mean([r["retrieval"]["rr"] for r in recs if r.get("retrieval")]),
            "precision_at_5": _mean([r["retrieval"]["precision"] for r in recs if r.get("retrieval")]),
            "judge_accuracy": _mean([GRADE_SCORE.get(g) for g in grades]),
            "judge_correct": grades.count("correct"),
            "judge_partial": grades.count("partial"),
            "judge_wrong": grades.count("wrong"),
            "bertscore_f1": _mean([r.get("bertscore_f1") for r in recs]),
            "false_refusals": sum(is_refusal(r) for r in recs),
            "out_of_scope_refused": sum(r["correct_refusal"] for r in oos),
            "out_of_scope_total": len(oos),
            "latency_ms": {
                "retrieve": timing("retrieve_ms"),
                "rerank": timing("rerank_ms"),
                "generate": timing("generate_ms", answered),
                "total": timing("total_ms", answered),
            },
        }
    return summary


def write_report(summary, in_scope, out_scope, versions, raw):
    def fmt(x, pct=False):
        if x is None:
            return "–"
        return f"{x * 100:.0f}%" if pct else f"{x:.3f}"

    lines = [
        "# Version comparison: " + " vs ".join(LABELS[v] for v in versions),
        "",
        f"Generated by `python eval/compare_versions.py` on {time.strftime('%Y-%m-%d %H:%M')}. "
        f"{len(in_scope)} paraphrased in-scope questions with gold labels and {len(out_scope)} "
        f"out-of-scope questions (`data/eval_questions.json`). V2-V4 share embeddings, chunks, prompt, "
        f"LLM (`openai/gpt-oss-120b`) and refusal threshold; only retrieval changes. V1 uses the same LLM "
        f"with no retrieval, so it has no retrieval scores and no sources. "
        f"Judge: `{JUDGE_MODEL}`.",
        "",
        "## Summary",
        "",
        "| Metric | " + " | ".join(summary[v]["label"] for v in versions) + " |",
        "|---|" + "---|" * len(versions),
    ]
    rows = [
        ("Retrieval: hit@5 (gold chunk in context)", lambda s: fmt(s["hit_at_5"], True)),
        ("Retrieval: MRR", lambda s: fmt(s["mrr"])),
        ("Retrieval: precision@5", lambda s: fmt(s["precision_at_5"], True)),
        ("Answer accuracy (judge, correct=1 partial=0.5)", lambda s: fmt(s["judge_accuracy"], True)),
        ("Judge grades correct / partial / wrong",
         lambda s: f"{s['judge_correct']} / {s['judge_partial']} / {s['judge_wrong']}"),
        ("BERTScore F1 vs gold answer", lambda s: fmt(s["bertscore_f1"])),
        ("In-scope questions refused (lower is better)", lambda s: f"{s['false_refusals']} / {s['in_scope_scored']}"),
        ("Out-of-scope correctly refused", lambda s: f"{s['out_of_scope_refused']} / {s['out_of_scope_total']}"),
        ("Retrieval time, median ms", lambda s: str(s["latency_ms"]["retrieve"]["median"] or "–")),
        ("Rerank time, median ms", lambda s: str(s["latency_ms"]["rerank"]["median"] or "–")),
        ("LLM generation time, median ms", lambda s: str(s["latency_ms"]["generate"]["median"])),
        ("Total response time, median ms", lambda s: str(s["latency_ms"]["total"]["median"])),
        ("Total response time, mean ms", lambda s: str(s["latency_ms"]["total"]["mean"])),
    ]
    for name, f in rows:
        lines.append(f"| {name} | " + " | ".join(f(summary[v]) for v in versions) + " |")

    lines += [
        "",
        "Response times include the Groq API call, so they vary with network and server load. "
        "Retrieval and rerank times are measured locally on CPU.",
        "",
        "## Per question",
        "",
        "Rank = position of the first gold chunk in the final context (– = not retrieved). "
        "Grade from the LLM judge.",
        "",
        "| ID | Question | " + " | ".join(f"{summary[v]['label']} rank / grade" for v in versions) + " |",
        "|---|---|" + "---|" * len(versions),
    ]
    for q in in_scope:
        cells = []
        for v in versions:
            rec = raw.get(f"{v}:{q['id']}") or {}
            if "error" in rec or not rec:
                cells.append("error")
                continue
            rank = (rec.get("retrieval") or {}).get("rank") or "–"
            grade = (rec.get("judge") or {}).get("grade") or "–"
            cells.append(f"{rank} / {grade}" + (" (refused)" if rec["refused"] else ""))
        lines.append(f"| {q['id']} | {q['question']} | " + " | ".join(cells) + " |")
    for q in out_scope:
        cells = []
        for v in versions:
            rec = raw.get(f"{v}:{q['id']}") or {}
            cells.append("error" if "error" in rec or not rec
                         else ("refused ✓" if rec["correct_refusal"] else "answered ✗"))
        lines.append(f"| {q['id']} | {q['question']} | " + " | ".join(cells) + " |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    OUT_JSON.write_text(json.dumps({"summary": summary, "judge_model": JUDGE_MODEL,
                                    "results": raw}, indent=2))
    print(f"\nWrote {OUT_MD} and {OUT_JSON}")
    print("\n".join(lines[6:8 + len(rows)]))


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versions", default="v1,v2,v3,v4")
    parser.add_argument("--limit", type=int, default=None, help="only the first N in-scope questions")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="ignore cached results")
    parser.add_argument("--pace", type=float, default=12.0,
                        help="seconds to wait after each LLM call (Groq free tier: 8k tokens/min)")
    args = parser.parse_args()

    versions = [v.strip() for v in args.versions.split(",")]
    data = json.loads(QUESTIONS_FILE.read_text())
    in_scope = data["in_scope"][: args.limit] if args.limit else data["in_scope"]
    out_scope = data["out_of_scope"]
    items = in_scope + out_scope

    qa = json.loads(QA_FILE.read_text())
    gold_answers = {
        q["id"]: [r["answer"] for r in qa if r["question"] in set(q["gold_questions"])]
        for q in in_scope
    }

    raw = load_raw(args.fresh)
    generate_answers(items, versions, raw, args.pace)

    for v in versions:
        for q in in_scope:
            rec = raw.get(f"{v}:{q['id']}")
            if rec and "error" not in rec:
                rec["retrieval"] = (retrieval_metrics(rec["sources"], q["gold_questions"])
                                    if v != "v1" else None)
        for q in out_scope:
            rec = raw.get(f"{v}:{q['id']}")
            if rec and "error" not in rec:
                rec["correct_refusal"] = is_refusal(rec)
    save_raw(raw)

    add_bertscore(in_scope, versions, raw, gold_answers)
    if not args.no_judge:
        judge_answers(in_scope, versions, raw, gold_answers, pace=args.pace / 2)
    send_feedback(items, versions, raw)

    summary = summarise(in_scope, out_scope, versions, raw)
    write_report(summary, in_scope, out_scope, versions, raw)


if __name__ == "__main__":
    main()

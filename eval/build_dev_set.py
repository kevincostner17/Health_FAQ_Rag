"""
Build a development set for tuning V4's reranker, kept separate from the test set
(data/eval_questions.json) so tuning choices are not made on the questions used to report results.

Samples MedQuAD questions from topics NOT used in the test set and asks a Groq model to rewrite
them the way a patient would ask. Gold label = the original MedQuAD question.

Run: python eval/build_dev_set.py
Output: data/dev_questions.json
"""
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

QA_FILE = REPO_ROOT / "data" / "qa_pairs.json"
TEST_FILE = REPO_ROOT / "data" / "eval_questions.json"
OUT_FILE = REPO_ROOT / "data" / "dev_questions.json"
N = 40
QTYPES = {"symptoms", "treatment", "exams and tests", "outlook", "stages", "susceptibility",
          "information", "prevention", "research"}

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite each medical FAQ question the way a worried patient or relative would type it "
               "into a chatbot: everyday words, lay names or abbreviations where natural (e.g. 'voice box' "
               "for larynx), sometimes a short personal context. Keep the same meaning and the same "
               "condition. Return JSON only: a list of strings, one per input, in the same order."),
    ("human", "{questions}"),
])


def main():
    qa = json.loads(QA_FILE.read_text())
    test = json.loads(TEST_FILE.read_text())
    test_gold = {g for q in test["in_scope"] for g in q["gold_questions"]}
    test_topics = {r["focus"] for r in qa if r["question"] in test_gold}

    pool = {}
    for r in qa:
        if r["focus"] not in test_topics and r["qtype"] in QTYPES:
            pool.setdefault(r["question"], r)
    sample = random.Random(7).sample(sorted(pool), N)

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, reasoning_effort="low",
                   max_tokens=4096, max_retries=8)
    rewritten = (PROMPT | llm | JsonOutputParser()).invoke(
        {"questions": json.dumps([q.rstrip(" ?") + "?" for q in sample], indent=1)}
    )
    if len(rewritten) != len(sample):
        raise RuntimeError(f"Expected {len(sample)} rewrites, got {len(rewritten)}")

    items = [{"id": f"d{i + 1:02d}", "question": new, "gold_questions": [orig]}
             for i, (orig, new) in enumerate(zip(sample, rewritten))]
    OUT_FILE.write_text(json.dumps({
        "description": "Development set for tuning the V4 reranker. Topics do not overlap the test set.",
        "in_scope": items,
    }, indent=2))
    print(f"Wrote {len(items)} dev questions to {OUT_FILE}")
    for it in items[:8]:
        print(f"  {it['gold_questions'][0]}  ->  {it['question']}")


if __name__ == "__main__":
    main()

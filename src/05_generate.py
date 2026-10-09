"""
Step 5: Generation
Retrieve relevant chunks for a question, then generate an answer using Groq.

Run: python src/05_generate.py
Requires: 03_embed_store.py to have been run first (creates chroma_db/)

SETUP (Groq):
1. Sign up free at https://console.groq.com
2. Create an API key
3. Add to .env: GROQ_API_KEY=your-key-here
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from rag_utils import format_context, is_likely_out_of_kb, load_collection, retrieve

TOP_K = 6
GROQ_MODEL = "openai/gpt-oss-120b"
CLOUDFLARE_MODEL = "@cf/tiiuae/falcon-7b-instruct"
GENERATION_BACKEND = "groq"

SYSTEM_PROMPT = """You are a careful health-information assistant. Answer ONLY using \
the provided context. If the context does not contain the answer, say clearly that \
you don't have enough information to answer, and suggest the user consult a medical \
professional. Do not speculate or use outside knowledge. Cite the source focus/topic \
for each claim where possible."""

OUT_OF_KB_REPLY = (
    "I don't have enough information in my health knowledge base to answer that question. "
    "Please consult a medical professional or a trusted health resource."
)


def generate_answer_groq(question: str, context: str) -> str:
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com, "
            "then create a .env file containing:\n  GROQ_API_KEY=your-key-here"
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content


def generate_answer_cloudflare(question: str, context: str) -> str:
    import requests

    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not api_token:
        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set."
        )

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{CLOUDFLARE_MODEL}"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success", False):
        raise RuntimeError(f"Cloudflare API error: {data.get('errors')}")
    return data["result"]["response"]


def generate_answer(question: str, context: str) -> str:
    if not context.strip():
        return OUT_OF_KB_REPLY
    if GENERATION_BACKEND == "cloudflare":
        return generate_answer_cloudflare(question, context)
    if GENERATION_BACKEND == "groq":
        return generate_answer_groq(question, context)
    raise ValueError(f"Unknown GENERATION_BACKEND: {GENERATION_BACKEND}")


def answer_question(question: str, collection) -> str:
    retrieved = retrieve(question, collection, top_k=TOP_K)
    context = format_context(retrieved)
    if is_likely_out_of_kb(retrieved):
        return OUT_OF_KB_REPLY
    return generate_answer(question, context)


def main():
    try:
        collection = load_collection()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    test_questions = [
        "What are the symptoms of leukemia?",
        "How is breast cancer treated?",
        "What are the treatments for Prostate Cancer ?",
        "What is the capital of France?",
    ]

    for question in test_questions:
        print("=" * 80)
        print(f"Q: {question}")
        try:
            answer = answer_question(question, collection)
            print(f"\nA: {answer}\n")
        except RuntimeError as e:
            print(f"\n[Generation skipped] {e}\n")


if __name__ == "__main__":
    main()

"""
Version 4 - RAG with reranking (deployed version)

Project Title:      Health FAQ RAG Chatbot (Group 63)
Problem Statement:  Cancer information online is scattered and unreliable; general chatbots can
                    invent medical facts with no sources.
Proposed Solution:  A RAG chatbot grounded in NIH Cancer.gov content, built up from V1 to V4.
Target Users:       Patients, carers and the public with general cancer questions.
Data/Documents:     MedQuAD CancerGov subset (729 NIH Q-A pairs, 98 cancer topics).
Expected Output:    A plain-language answer with sources, or a refusal when out of scope.
(Full text: versions/common.py)

Flow:
    User query -> Chroma retriever (top 20, cosine >= 0.45) -> reranker (cross-encoder)
    -> top 5 -> prompt + context -> LLM -> answer

Why rerank: the embedding model encodes the question and each chunk separately, so it only
compares two summaries. A cross-encoder reads the question and the chunk together and scores
how well that chunk answers that question. It is too slow to run on all 6,452 chunks, so the
retriever narrows to 20 candidates first and the cross-encoder picks the best 5.

Run: python versions/v4_rerank.py ["your question"]
"""
import sys
import time
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from versions.common import FINAL_K, ScoredRetriever, build_rag_chain, print_result, run_chain
from versions.v3_chroma import get_vectorstore

VERSION = "v4"
CANDIDATES_K = 20
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


def rerank_documents(question: str, docs: list[Document], top_n: int = FINAL_K) -> list[Document]:
    """Score each (question, chunk) pair with the cross-encoder and keep the best top_n."""
    if not docs:
        return []
    scores = get_cross_encoder().predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)[:top_n]
    return [
        Document(
            id=doc.id,
            page_content=doc.page_content,
            metadata={**doc.metadata, "rerank_score": round(float(score), 3)},
        )
        for doc, score in ranked
    ]


def _rerank_step(state: dict) -> dict:
    start = time.perf_counter()
    docs = rerank_documents(state["question"], state["docs"])
    rerank_ms = round((time.perf_counter() - start) * 1000, 1)
    return {**state, "docs": docs, "timings": {**state["timings"], "rerank_ms": rerank_ms}}


reranker = RunnableLambda(_rerank_step, name="reranker")


@lru_cache(maxsize=1)
def build_chain():
    get_cross_encoder()  # load the model now so the first question's rerank time is fair
    retriever = ScoredRetriever(vectorstore=get_vectorstore(), k=CANDIDATES_K)
    return build_rag_chain(retriever, run_name="V4 Rerank RAG", reranker=reranker)


def ask(question: str, config: dict | None = None) -> dict:
    return run_chain(build_chain(), question, VERSION, config)


def main():
    questions = sys.argv[1:] or [
        "How do they treat cancer of the voice box?",
        "Who won the FIFA World Cup in 2022?",
    ]
    for q in questions:
        print_result(ask(q))


if __name__ == "__main__":
    main()

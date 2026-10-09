"""
Version 2 - Basic RAG with FAISS

Project Title:      Health FAQ RAG Chatbot (Group 63)
Problem Statement:  Cancer information online is scattered and unreliable; general chatbots can
                    invent medical facts with no sources.
Proposed Solution:  A RAG chatbot grounded in NIH Cancer.gov content, built up from V1 to V4.
Target Users:       Patients, carers and the public with general cancer questions.
Data/Documents:     MedQuAD CancerGov subset (729 NIH Q-A pairs, 98 cancer topics).
Expected Output:    A plain-language answer with sources, or a refusal when out of scope.
(Full text: versions/common.py)

Flow:
    Documents -> MedQuADLoader -> RecursiveCharacterTextSplitter -> MiniLM embeddings
    -> FAISS index -> ScoredRetriever (top 5, cosine >= 0.45) -> prompt + context -> LLM -> answer

FAISS is an in-memory similarity-search library. The index is saved to faiss_index/ as two
files (vectors + a pickled docstore). It has no database server, no metadata filtering and
no incremental updates beyond rebuilding or appending in code.

Run: python versions/v2_faiss.py ["your question"]
"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from versions.common import (
    FAISS_DIR,
    FINAL_K,
    ScoredRetriever,
    build_rag_chain,
    get_embeddings,
    load_chunks,
    print_result,
    run_chain,
)

VERSION = "v2"


def _cosine_relevance(similarity: float) -> float:
    # Inner product of normalised vectors already is cosine similarity. LangChain's default
    # inner-product function returns 1 - similarity, which would invert the ranking scores.
    return similarity


def _faiss_kwargs():
    from langchain_community.vectorstores.utils import DistanceStrategy

    return {
        "distance_strategy": DistanceStrategy.MAX_INNER_PRODUCT,
        "relevance_score_fn": _cosine_relevance,
    }


def build_index():
    from langchain_community.vectorstores import FAISS

    chunks = load_chunks()
    print(f"V2: embedding {len(chunks)} chunks into FAISS ...")
    store = FAISS.from_documents(chunks, get_embeddings(), **_faiss_kwargs())
    store.save_local(str(FAISS_DIR))
    print(f"V2: saved FAISS index to {FAISS_DIR}")
    return store


@lru_cache(maxsize=1)
def get_vectorstore():
    from langchain_community.vectorstores import FAISS

    if not (FAISS_DIR / "index.faiss").exists():
        return build_index()
    # The docstore pickle is created by build_index() in this repo, so loading it is safe.
    return FAISS.load_local(
        str(FAISS_DIR), get_embeddings(), allow_dangerous_deserialization=True, **_faiss_kwargs()
    )


@lru_cache(maxsize=1)
def build_chain():
    retriever = ScoredRetriever(vectorstore=get_vectorstore(), k=FINAL_K)
    return build_rag_chain(retriever, run_name="V2 FAISS RAG")


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

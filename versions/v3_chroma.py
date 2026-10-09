"""
Version 3 - RAG with ChromaDB and metadata

Project Title:      Health FAQ RAG Chatbot (Group 63)
Problem Statement:  Cancer information online is scattered and unreliable; general chatbots can
                    invent medical facts with no sources.
Proposed Solution:  A RAG chatbot grounded in NIH Cancer.gov content, built up from V1 to V4.
Target Users:       Patients, carers and the public with general cancer questions.
Data/Documents:     MedQuAD CancerGov subset (729 NIH Q-A pairs, 98 cancer topics).
Expected Output:    A plain-language answer with sources, or a refusal when out of scope.
(Full text: versions/common.py)

Flow:
    Documents -> chunking -> MiniLM embeddings -> ChromaDB (cosine)
    -> ScoredRetriever (top 5, cosine >= 0.45) -> prompt + context -> LLM -> answer

Metadata stored with every chunk:
    source      NIH Cancer.gov URL of the original document
    page        section number of the chunk within its answer (MedQuAD has no printed pages)
    doc_type    question type: symptoms, treatment, stages, outlook, ...
    topic, question, qid, total_pages

FAISS vs ChromaDB:
    FAISS is a similarity-search library: vectors live in memory and are saved as files, and
    any metadata sits in a separate Python docstore. ChromaDB is a vector database: it persists
    vectors, text and metadata together, supports metadata filters (e.g. doc_type="treatment"),
    adding/deleting single records, and can run as a server. For 6,452 chunks both are fast;
    Chroma's value is data management, not raw speed.

Run: python versions/v3_chroma.py ["your question"]
"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from versions.common import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    FINAL_K,
    ScoredRetriever,
    build_rag_chain,
    get_embeddings,
    load_chunks,
    print_result,
    run_chain,
)

VERSION = "v3"
BATCH = 1000


@lru_cache(maxsize=1)
def get_vectorstore():
    from langchain_chroma import Chroma

    store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    if store._collection.count() == 0:
        chunks = load_chunks()
        print(f"V3: embedding {len(chunks)} chunks into Chroma ...")
        for i in range(0, len(chunks), BATCH):
            store.add_documents(chunks[i : i + BATCH])
        print(f"V3: stored {store._collection.count()} chunks in {CHROMA_DIR}")
    return store


@lru_cache(maxsize=1)
def build_chain():
    retriever = ScoredRetriever(vectorstore=get_vectorstore(), k=FINAL_K)
    return build_rag_chain(retriever, run_name="V3 Chroma RAG")


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

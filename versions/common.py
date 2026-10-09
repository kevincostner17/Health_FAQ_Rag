"""
Project Title:
    Health FAQ RAG Chatbot (Group 63)

Problem Statement:
    People looking up cancer information online face scattered, inconsistent and sometimes
    unreliable sources. A general-purpose chatbot can answer fluently but may invent medical
    facts and gives no way to check where an answer came from.

Proposed Solution:
    A Retrieval-Augmented Generation (RAG) chatbot that answers only from trusted NIH Cancer.gov
    content. Questions are matched to the most relevant passages, an LLM writes the answer from
    those passages alone, sources are shown with every answer, and out-of-scope questions are
    refused. The system is built in four versions (V1 LangChain basics, V2 FAISS, V3 ChromaDB,
    V4 reranking) so each improvement can be measured.

Target Users:
    Patients, carers and members of the public with general questions about cancer types,
    symptoms, risk factors, diagnosis, stages, treatment and outlook. Not a substitute for
    a doctor.

Data/Documents to be used:
    MedQuAD (Medical Question Answering Dataset), CancerGov subset from the US National
    Institutes of Health: 729 question-answer pairs covering 98 cancer topics
    (data/qa_pairs.json), split into 6,452 chunks.

Expected Output:
    A short, plain-language answer grounded in the retrieved NIH passages, the list of source
    documents (topic, URL, section/page, document type) used to write it, or a clear refusal
    when the question is outside the knowledge base.

---------------------------------------------------------------------------------------------
Shared building blocks for versions V1-V4: document loader, splitter, embeddings, LLM, prompt,
retriever and the RAG chain used by V2, V3 and V4.
"""
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

QA_FILE = REPO_ROOT / "data" / "qa_pairs.json"
FAISS_DIR = REPO_ROOT / "faiss_index"
CHROMA_DIR = REPO_ROOT / "chroma_lc"
CHROMA_COLLECTION = "medquad_lc"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
FINAL_K = 5
# Cosine similarity: 1 = identical, 0 = unrelated. Chunks below this are dropped.
# Equals the original pipeline's cosine distance limit of 0.55.
MIN_RELEVANCE = 0.45

SYSTEM_PROMPT = """You are a careful health-information assistant. Answer ONLY using \
the provided context. If the context does not contain the answer, say clearly that \
you don't have enough information to answer, and suggest the user consult a medical \
professional. Do not speculate or use outside knowledge. Cite the source focus/topic \
for each claim where possible."""

OUT_OF_KB_REPLY = (
    "I don't have enough information in my health knowledge base to answer that question. "
    "Please consult a medical professional or a trusted health resource."
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


class MedQuADLoader(BaseLoader):
    """Loads MedQuAD question-answer records as one Document per answer."""

    def __init__(self, path: Path = QA_FILE):
        self.path = Path(path)

    def lazy_load(self):
        with open(self.path, encoding="utf-8") as f:
            records = json.load(f)
        for idx, rec in enumerate(records):
            yield Document(
                page_content=re.sub(r"\s+", " ", rec["answer"]).strip(),
                metadata={
                    "record": idx,
                    "source": rec["url"],
                    "topic": rec["focus"],
                    "doc_type": rec["qtype"] or "general",
                    "question": rec["question"],
                    "qid": rec["qid"],
                },
            )


def split_documents(docs: list[Document]) -> list[Document]:
    """Chunk each answer; number the chunks as pages and prefix topic + question."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc.page_content)
        for i, piece in enumerate(pieces):
            meta = doc.metadata
            chunks.append(Document(
                id=f"rec{meta['record']}_p{i + 1}",
                page_content=f"Topic: {meta['topic']}. Question: {meta['question']}. Answer: {piece}",
                metadata={**meta, "page": i + 1, "total_pages": len(pieces), "chunk_text": piece},
            ))
    return chunks


def load_chunks() -> list[Document]:
    return split_documents(MedQuADLoader().load())


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_llm():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com, "
            "then create a .env file containing:\n  GROQ_API_KEY=your-key-here"
        )
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        reasoning_effort="low",
        max_tokens=1024,
        max_retries=8,
    )


class ScoredRetriever(BaseRetriever):
    """Top-k similarity search that drops weak matches and records each chunk's score."""

    vectorstore: VectorStore
    k: int = FINAL_K
    min_relevance: float = MIN_RELEVANCE

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.k)
        return [
            Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata={**doc.metadata, "similarity": round(float(score), 4)},
            )
            for doc, score in results
            if score >= self.min_relevance
        ]


def format_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata['topic']} | {d.metadata['doc_type']} | page {d.metadata['page']}]\n"
        f"{d.metadata['chunk_text']}"
        for d in docs
    )


def to_sources(docs: list[Document]) -> list[dict]:
    sources = []
    for d in docs:
        m = d.metadata
        source = {
            "focus": m["topic"],
            "source": "CancerGov",
            "url": m["source"],
            "question": m["question"],
            "doc_type": m["doc_type"],
            "page": m["page"],
            "total_pages": m["total_pages"],
            "chunk": m["chunk_text"],
            "similarity": m.get("similarity"),
            "distance": round(1 - m["similarity"], 3) if m.get("similarity") is not None else None,
        }
        if "rerank_score" in m:
            source["rerank_score"] = m["rerank_score"]
        sources.append(source)
    return sources


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def build_rag_chain(retriever: BaseRetriever, run_name: str, reranker=None):
    """retrieve -> [rerank] -> generate, passing a state dict between named steps."""

    def retrieve(state: dict, config) -> dict:
        start = time.perf_counter()
        docs = retriever.invoke(state["question"], config=config)
        return {**state, "docs": docs, "timings": {"retrieve_ms": _ms(start)}}

    def generate(state: dict, config) -> dict:
        start = time.perf_counter()
        docs = state["docs"]
        if docs:
            answer_chain = (RAG_PROMPT | get_llm() | StrOutputParser()).with_config(run_name="llm_answer")
            answer = answer_chain.invoke(
                {"context": format_context(docs), "question": state["question"]}, config=config
            )
        else:
            answer = OUT_OF_KB_REPLY
        timings = {**state["timings"], "generate_ms": _ms(start)}
        return {
            "question": state["question"],
            "answer": answer,
            "refused": not docs,
            "sources": to_sources(docs),
            "timings": timings,
        }

    chain = RunnableLambda(retrieve, name="retrieve")
    if reranker is not None:
        chain = chain | reranker
    chain = chain | RunnableLambda(generate, name="generate")
    return chain.with_config(run_name=run_name)


def run_chain(chain, question: str, version: str, config: dict | None = None) -> dict:
    """Invoke a V2-V4 chain with tracing tags and total latency."""
    config = dict(config or {})
    config["tags"] = [version, *config.get("tags", [])]
    config["metadata"] = {"version": version, **config.get("metadata", {})}
    start = time.perf_counter()
    result = chain.invoke({"question": question.strip()}, config=config)
    result["timings"]["total_ms"] = _ms(start)
    return result


def print_result(result: dict) -> None:
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    for i, s in enumerate(result["sources"], 1):
        extra = f", rerank {s['rerank_score']:.2f}" if "rerank_score" in s else ""
        print(f"   [{i}] {s['focus']} | {s['doc_type']} | page {s['page']}/{s['total_pages']} "
              f"| sim {s['similarity']}{extra}")
    print(f"   timings: {result['timings']}")

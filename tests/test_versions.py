"""Unit tests for the V1-V4 building blocks (no Groq calls). Run: python -m pytest tests"""
import json
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from versions import common


@pytest.fixture(scope="module")
def chunks():
    return common.load_chunks()


def test_loader_reads_every_record():
    docs = common.MedQuADLoader().load()
    records = json.loads(common.QA_FILE.read_text())
    assert len(docs) == len(records)
    assert "  " not in docs[0].page_content
    assert set(docs[0].metadata) >= {"source", "topic", "doc_type", "question", "qid"}


def test_chunks_match_original_pipeline_and_carry_metadata(chunks):
    assert len(chunks) == 6452  # same as src/02_chunk.py
    first = chunks[0]
    assert first.metadata["page"] == 1
    assert first.metadata["total_pages"] >= 1
    assert first.metadata["source"].startswith("http")
    assert first.page_content.startswith(f"Topic: {first.metadata['topic']}. Question:")
    assert len(first.metadata["chunk_text"]) <= common.CHUNK_SIZE
    assert len({c.id for c in chunks}) == len(chunks)


def test_faiss_and_chroma_scores_are_on_the_same_cosine_scale():
    from versions import v2_faiss, v3_chroma

    q = "What are the treatments for Laryngeal Cancer ?"
    faiss_top = v2_faiss.get_vectorstore().similarity_search_with_relevance_scores(q, k=3)
    chroma_top = v3_chroma.get_vectorstore().similarity_search_with_relevance_scores(q, k=3)
    assert faiss_top[0][0].id == chroma_top[0][0].id
    assert faiss_top[0][1] == pytest.approx(chroma_top[0][1], abs=1e-3)
    assert faiss_top[0][1] > common.MIN_RELEVANCE
    assert faiss_top[0][1] >= faiss_top[-1][1]  # higher score = more similar


def test_retriever_drops_off_topic_questions():
    from versions import v3_chroma

    retriever = common.ScoredRetriever(vectorstore=v3_chroma.get_vectorstore())
    assert retriever.invoke("Who won the FIFA World Cup in 2022?") == []
    docs = retriever.invoke("What are the symptoms of melanoma?")
    assert docs and all(d.metadata["similarity"] >= common.MIN_RELEVANCE for d in docs)


def test_reranker_orders_by_score_and_trims():
    from versions.v4_rerank import rerank_documents

    def doc(text, i):
        return Document(id=str(i), page_content=text, metadata={"i": i})

    docs = [doc("Bananas are a good source of potassium.", i) for i in range(6)]
    docs.append(doc("Laryngeal cancer is treated with surgery, radiation therapy and chemotherapy.", 99))
    ranked = rerank_documents("How is laryngeal cancer treated?", docs, top_n=3)
    assert len(ranked) == 3
    assert ranked[0].metadata["i"] == 99
    scores = [d.metadata["rerank_score"] for d in ranked]
    assert scores == sorted(scores, reverse=True)
    assert rerank_documents("anything", []) == []


def test_chain_refuses_without_calling_llm_when_nothing_is_retrieved(monkeypatch):
    from langchain_core.retrievers import BaseRetriever

    class EmptyRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            return []

    def no_llm():
        raise AssertionError("LLM must not be called for a refusal")

    monkeypatch.setattr(common, "get_llm", no_llm)
    chain = common.build_rag_chain(EmptyRetriever(), run_name="test")
    result = common.run_chain(chain, "  off topic  ", "test")
    assert result["refused"] is True
    assert result["answer"] == common.OUT_OF_KB_REPLY
    assert result["sources"] == []
    assert set(result["timings"]) == {"retrieve_ms", "generate_ms", "total_ms"}


def test_retrieval_metrics():
    from compare_versions import retrieval_metrics

    sources = [{"question": "other"}, {"question": "gold"}, {"question": "gold"}]
    m = retrieval_metrics(sources, ["gold"])
    assert m == {"hit": True, "rank": 2, "rr": 0.5, "precision": pytest.approx(2 / 3)}
    miss = retrieval_metrics([{"question": "other"}], ["gold"])
    assert miss["hit"] is False and miss["rr"] == 0.0


def test_v1_parser_accepts_json_and_falls_back_to_plain_text():
    from versions.v1_basics import parse_answer

    parsed = parse_answer('{"answer": "a", "key_points": ["b"], "disclaimer": "c"}')
    assert parsed.answer == "a" and parsed.key_points == ["b"]
    fallback = parse_answer("Sorry, I can only help with health questions.")
    assert fallback.answer.startswith("Sorry") and fallback.key_points == []

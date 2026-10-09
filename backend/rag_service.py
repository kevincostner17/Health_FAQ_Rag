"""RAG pipeline service for the API layer (uses V4: Chroma retrieval + cross-encoder reranking)."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from backend.ragas_metrics import compute_metrics
from versions import v4_rerank


def get_pipeline():
    """Load the embedding model, Chroma store and reranker once (built on first use)."""
    return v4_rerank.build_chain()


def ask(question: str, compute_ragas: bool = True) -> dict:
    result = v4_rerank.ask(question, config={"tags": ["api"]})

    metrics = (
        compute_metrics(question, result["answer"], result["sources"], result["refused"])
        if compute_ragas
        else {"faithfulness": None, "relevance": None, "hallucination": None, "method": "skipped"}
    )

    return {
        "question": result["question"],
        "answer": result["answer"],
        "refused": result["refused"],
        "sources": result["sources"],
        "metrics": metrics,
    }

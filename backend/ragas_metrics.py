"""RAGAS-based response quality metrics: faithfulness, relevance, hallucination."""
import logging
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from metrics import looks_like_refusal

logger = logging.getLogger(__name__)

# Inline citation tags such as 【Source: Topic | CancerGov】 are formatting, not claims;
# left in, the judge tries to verify them and under-scores correct answers.
CITATION_PATTERN = re.compile(r"【[^】]*】|\[Source:[^\]]*\]")

# Judge LLM on Groq: a smaller model than the generator (openai/gpt-oss-120b), so answers
# are not graded by the model that wrote them. Qwen 3 is not usable on the free tier
# (1,000 output tokens/min). Override with RAGAS_JUDGE_MODEL in .env.
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "openai/gpt-oss-20b")
# answer_relevancy needs embeddings; reuse the local retrieval model (no OpenAI key needed).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_judge = None
_embeddings = None


def _get_judge_and_embeddings(api_key: str):
    global _judge, _embeddings
    if _judge is None:
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        # Groq rejects n > 1, so bypass_n makes RAGAS issue one completion per call.
        # Low reasoning effort + generous max_tokens stop long answers being cut off.
        _judge = LangchainLLMWrapper(
            ChatGroq(
                model=JUDGE_MODEL,
                api_key=api_key,
                temperature=0,
                reasoning_effort="low",
                max_tokens=4096,
            ),
            bypass_n=True,
        )
        _embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    return _judge, _embeddings


def _contexts_from_sources(sources: list[dict]) -> list[str]:
    return [s.get("chunk", "") for s in sources if s.get("chunk")]


def _heuristic_metrics(question: str, answer: str, contexts: list[str], refused: bool) -> dict:
    if refused or not contexts:
        return {
            "faithfulness": 1.0,
            "relevance": 1.0 if refused else 0.0,
            "hallucination": 0.0,
            "method": "heuristic_refusal",
        }
    ctx_words = set(" ".join(contexts).lower().split())
    ans_words = set(answer.lower().split())
    overlap = len(ctx_words & ans_words) / max(len(ans_words), 1)
    faithfulness = min(1.0, 0.35 + overlap * 0.65)
    relevance = min(1.0, 0.4 + overlap * 0.5)
    return {
        "faithfulness": round(faithfulness, 3),
        "relevance": round(relevance, 3),
        "hallucination": round(max(0.0, 1.0 - faithfulness), 3),
        "method": "heuristic",
    }


def compute_metrics(
    question: str, answer: str, sources: list[dict], refused: bool, patient: bool = False
) -> dict:
    """
    Compute Faithfulness, Relevance (answer_relevancy), and Hallucination score.

    Hallucination is defined as (1 - faithfulness): higher = more likely hallucinated.
    Uses RAGAS (Groq judge + local embeddings) when GROQ_API_KEY is available; falls back
    to heuristics on failure, with the reason in metrics["error"].

    patient=True (batch eval) runs judge calls one at a time and retries through Groq
    free-tier rate limits (8,000 tokens/min); live chat fails fast instead.
    """
    contexts = _contexts_from_sources(sources)
    if refused:
        return {
            "faithfulness": 1.0,
            "relevance": 1.0,
            "hallucination": 0.0,
            "method": "refusal",
        }
    if not contexts:
        return _heuristic_metrics(question, answer, contexts, refused=True)
    if looks_like_refusal(answer):
        # The model declined despite retrieved context: no unsupported claims, but not an
        # answer either, so relevance is left empty rather than scored as a perfect 1.0.
        return {
            "faithfulness": 1.0,
            "relevance": None,
            "hallucination": 0.0,
            "method": "model_refusal",
        }
    judged_answer = CITATION_PATTERN.sub("", answer).strip()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _heuristic_metrics(question, answer, contexts, refused)

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
        from ragas.run_config import RunConfig

        llm, embeddings = _get_judge_and_embeddings(api_key)
        ds = Dataset.from_dict({
            "question": [question],
            "answer": [judged_answer],
            "contexts": [contexts],
        })
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=True,
            show_progress=False,
            run_config=(
                RunConfig(timeout=900, max_retries=10, max_wait=60, max_workers=1)
                if patient
                else RunConfig(timeout=90, max_retries=2)
            ),
        )
        df = result.to_pandas()
        faith = float(df["faithfulness"].iloc[0]) if "faithfulness" in df.columns else None
        rel = float(df["answer_relevancy"].iloc[0]) if "answer_relevancy" in df.columns else None
        if faith is None or rel is None or math.isnan(faith) or math.isnan(rel):
            raise ValueError("RAGAS returned incomplete metrics")
        faith = max(0.0, min(1.0, faith))
        rel = max(0.0, min(1.0, rel))
        return {
            "faithfulness": round(faith, 3),
            "relevance": round(rel, 3),
            "hallucination": round(max(0.0, 1.0 - faith), 3),
            "method": "ragas",
        }
    except Exception as e:
        # Fall back so the chat still works, but record why RAGAS failed instead of hiding it.
        logger.warning("RAGAS scoring failed, using heuristic: %s: %s", type(e).__name__, e)
        metrics = _heuristic_metrics(question, answer, contexts, refused)
        metrics["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return metrics

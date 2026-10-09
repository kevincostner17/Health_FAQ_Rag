# V1–V4 LangChain RAG versions — design

Date: 2026-10-09 · Branch: `feature/v1-v4-langchain-rag`

## Goal

Meet the course brief (V1 LangChain basics → V2 FAISS → V3 ChromaDB → V4 reranking, 20+ question
comparison, LangSmith monitoring, deployable V4 app) on top of the existing Health FAQ RAG project,
and be able to show *why* each version is better.

## Decisions

| Topic | Decision |
|---|---|
| Relation to existing code | New `versions/` package alongside `src/`; existing scripts and `chroma_db/` untouched. FastAPI and Streamlit switch to V4. |
| Approach | LangChain retrievers + hand-written reranker `RunnableLambda`; one LCEL chain per version. |
| Data | `data/qa_pairs.json` (MedQuAD CancerGov, 729 answers) via a custom `MedQuADLoader(BaseLoader)`. |
| Chunking | `RecursiveCharacterTextSplitter`, 500 chars / 75 overlap (same as current pipeline). Chunk text carries a `Topic: … Question: … Answer: …` prefix. |
| Metadata | `source` (NIH URL), `page` (section number within the answer), `total_pages`, `doc_type` (qtype), `topic`, `question`, `qid`. |
| Embeddings | `all-MiniLM-L6-v2`, normalised (cosine). |
| LLM | Groq `openai/gpt-oss-120b`, temperature 0. |
| V1 | `prompt | ChatGroq | PydanticOutputParser` → `{answer, key_points, disclaimer}`; no retrieval. |
| V2 | FAISS (`IndexFlatIP` on normalised vectors = cosine), top 5. |
| V3 | Chroma (`hnsw:space = cosine`), top 5, metadata shown with sources. |
| V4 | Chroma top 20 → cross-encoder `ms-marco-MiniLM-L-6-v2` → top 5. |
| Refusal gate | Same in V2–V4: drop chunks with cosine relevance < 0.45 (= current distance > 0.55). No chunks left → fixed refusal, LLM not called. Reranker only reorders/trims. |
| Indexes | `faiss_index/`, `chroma_lc/` (gitignored), built by `python run.py build-versions` or on first use. |
| LangSmith | `LANGSMITH_TRACING/API_KEY/PROJECT` env vars; run names `V1`…`V4`, tags and question metadata; judge scores attached as feedback. Off when no key. |

## Chain shape (V2–V4)

```
{"question"} → retrieve (retriever.invoke) → [V4: reranker] → generate (gate → prompt | llm | parser)
→ {question, answer, refused, sources[], timings{retrieve_ms, rerank_ms, generate_ms, total_ms}}
```

Every version exposes `build_chain()` and `ask(question, config=None)`; each file runs as a demo script.

## Evaluation

* `data/eval_questions.json`: 22 paraphrased in-scope questions (lay wording, abbreviations, symptom
  descriptions) each labelled with the MedQuAD question(s) whose answer is the gold answer, plus
  5 out-of-scope questions.
* Retrieval quality: hit@5, MRR, precision@5 — a retrieved chunk is relevant if its original
  MedQuAD question is one of the gold questions.
* Answer accuracy: BERTScore F1 vs the best-matching gold answer, and a Groq judge
  (`openai/gpt-oss-20b`) grading correct / partial / wrong (1 / 0.5 / 0) with a reason.
* Refusals: out-of-scope refusal rate and in-scope false-refusal rate.
* Response time: per-stage and total latency, mean and median.
* Questions are interleaved across versions (q1: V2, V3, V4; q2: …) so network drift affects all
  versions equally. Results are cached per (version, question) so a rate-limited run can resume.
* Output: `data/version_comparison.json` and `docs/version_comparison.md`.

## Error handling

Missing `GROQ_API_KEY` → clear `RuntimeError` (API 503, Streamlit `st.error`); missing indexes →
built automatically; Groq rate limits → client retries with backoff; question length 1–500 (existing).

## Testing

pytest unit tests for the loader/metadata, FAISS and Chroma score scale agreement, the reranker
ordering/trim, the refusal gate, and the retrieval metrics; end-to-end check by running each version
script and the comparison.

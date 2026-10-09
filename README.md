# Health FAQ RAG Chatbot

**Group ID:** 63  
**Project Name:** Health FAQ RAG Chatbot

## Project Overview

This project develops a Retrieval-Augmented Generation (RAG) FAQ chatbot for consumer health questions, grounded in trusted NIH health information. It is built as part of the WIL Project (Test-Driven RAG), reproducing the RAG pipeline and evaluation methodology from [Walert](https://github.com/rmit-ir/walert).

The goal is to help everyday users get fast, reliable answers to health questions instead of navigating fragmented or low-quality sources, while measuring the system's effectiveness, faithfulness, and correct refusal behaviour on out-of-scope questions.

## Knowledge Base

We use **MedQuAD** (Medical Question Answering Dataset), scoped to the **CancerGov** subset from NIH. Our working set contains **729 question-answer pairs** across **98 cancer topics**, chunked into **6,452 retrieval units**.

## Pipeline Architecture

Following Walert's retrieval → index → generation approach:

1. **Source & scope** — parse MedQuAD CancerGov XML into question-answer-source records (`01_ingest.py`)
2. **Chunk** — split answers into 500-character chunks with 75-character overlap (`02_chunk.py`)
3. **Embed & store** — sentence-transformer embeddings stored in Chroma (`03_embed_store.py`)
4. **Retrieve** — semantic search with distance threshold for out-of-KB detection (`rag_utils.py`)
5. **Generate** — LLM answer conditioned on retrieved context (`05_generate.py`)

**Stack:**
- Orchestration: LangChain (text splitters)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: Chroma (persistent, cosine similarity); FAISS in V2
- Reranker (V4): `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Monitoring: LangSmith tracing
- LLM: Groq API (`openai/gpt-oss-120b`)
- Backend API: FastAPI + SQLite chat history
- Frontend: React (Vite) — healthcare-themed UI
- Quality metrics: RAGAS (faithfulness, answer relevancy, hallucination = 1 − faithfulness)

## Evaluation Framework

Test questions mirror Walert's three categories:

| Category | What it tests | Metric |
|---|---|---|
| Known | Answer exists in the KB | ROUGE-1 / BERTScore vs. gold answer |
| Inferred | Requires synthesis across chunks | Correctness / non-refusal rate |
| Out-of-KB | No answer in the KB | % correctly refused |

**RAGAS metrics (per response & batch):**
- **Faithfulness** — answer grounded in retrieved context
- **Relevance** — answer addresses the question (RAGAS `answer_relevancy`)
- **Hallucination** — `1 − faithfulness` (lower is better)

**Latest quick-eval results:** 80% known answered, avg BERTScore F1 0.849, 100% out-of-KB refusal.

## Repository Structure

```
├── data/              # Processed QA pairs, chunks, test sets, eval results
├── backend/           # FastAPI, RAGAS metrics, chat history (SQLite)
├── frontend/          # React healthcare UI (Vite)
├── src/               # Original pipeline scripts, RAG utils, Streamlit app, eval runner
├── versions/          # V1-V4 LangChain versions (V4 powers the apps)
├── tests/             # Unit tests for the V1-V4 building blocks
├── eval/              # Test set builder, metrics, RAGAS batch eval
├── docs/              # Pipeline explainer PDF, version comparison, design notes
├── chroma_db/         # Local vector store (generated, gitignored)
├── faiss_index/       # V2 FAISS index (generated, gitignored)
├── chroma_lc/         # V3/V4 Chroma store (generated, gitignored)
└── run.py             # Unified local runner
```

## Project Links

- **Trello board:** https://trello.com/b/healthfaqrag
- **GitHub repo:** https://github.com/kevincostner17/Health_FAQ_Rag

## Setup on a new Mac (step by step)

Do steps 1–4 **once**. After that, use "Run the app" every time.

### 1. Install the tools (once per Mac)

Open **Terminal** and install [Homebrew](https://brew.sh) if you don't have it:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install Python 3.12 and Node.js:

```bash
brew install python@3.12 node
```

> Use `python3.12`, not `python3`: the built-in `python3` on a new Mac is version 3.9, which is too old for this project.

### 2. Get the code

```bash
git clone https://github.com/kevincostner17/Health_FAQ_Rag.git
cd Health_FAQ_Rag
```

### 3. Add your Groq API key

Get a free key at <https://console.groq.com/keys>, then:

```bash
cp .env.example .env
open -e .env
```

Replace `your-key-here` with your key, save and close. (`.env` is private and is never uploaded to GitHub.)

### 4. Install and build everything

```bash
python3.12 run.py setup
```

This creates a `.venv` folder, installs the Python packages, builds the search database (`chroma_db/`) and the test sets. **The first run takes about 5–10 minutes** (it downloads PyTorch and the MiniLM embedding model).

## Run the app

Open **two Terminal windows** in the `Health_FAQ_Rag` folder.

**Terminal 1 — backend (API):**

```bash
python3.12 run.py api
```

Wait until you see `Application startup complete`.

**Terminal 2 — website:**

```bash
python3.12 run.py web
```

The first time, this also installs the website packages (about 1 minute). Then open **<http://localhost:5173>** in your browser and ask a question, e.g. *"What are the symptoms of leukemia?"*

To stop either one, press `Ctrl + C` in its Terminal window.

### Other commands

```bash
python3.12 run.py chat        # chat in the terminal (no website)
python3.12 run.py pipeline    # quick retrieval + generation test
python3.12 run.py eval        # full evaluation (ROUGE-1, BERTScore, refusal rate, RAGAS)
python3.12 run.py ragas       # batch RAGAS report -> data/ragas_results.json
```

The React app sends `/api/*` requests to the backend. Each chat is saved in `data/chat_history.db` with its faithfulness, relevance and hallucination scores.

### Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: python3.12` | Run step 1 again: `brew install python@3.12` |
| `command not found: npm` | `brew install node` |
| `GROQ_API_KEY not set` | Check that `.env` exists in the project folder and contains your key (step 3) |
| `Vector store not found` | Run `python3.12 run.py setup` |
| `Address already in use` (port 8000) | Another app is using port 8000. Start the backend on another port and point the website at it (two terminals): `.venv/bin/uvicorn backend.api:app --port 8001` and `cd frontend && VITE_API_URL=http://127.0.0.1:8001 npm run dev` |
| Answers say "Estimated from word overlap — RAGAS unavailable" | The free Groq plan has a per-minute/day limit. Answers still work; wait a minute for live RAGAS scores |
| Want a clean reinstall | `rm -rf .venv chroma_db frontend/node_modules` then repeat step 4 |

## RAG versions V1 → V4 (LangChain)

The `versions/` folder rebuilds the chatbot step by step with LangChain, as required by the RAG project brief. Every version shares the same data, chunks (500 characters, 75 overlap), embeddings (MiniLM), prompt and LLM (Groq `gpt-oss-120b`), so each step changes one thing. The project problem statement (title, problem, solution, target users, data, expected output) is at the top of [versions/common.py](versions/common.py) and summarised in every version file.

| Version | File | Flow | What it adds |
|---|---|---|---|
| V1 | [v1_basics.py](versions/v1_basics.py) | input → prompt → LLM → Pydantic parser | LangChain basics: `ChatPromptTemplate`, `ChatGroq`, `PydanticOutputParser`, LCEL chain and runnables. No documents, so no sources. |
| V2 | [v2_faiss.py](versions/v2_faiss.py) | loader → splitter → embeddings → **FAISS** → retriever (top 5) → LLM | Basic RAG: answers grounded in NIH text, sources shown, off-topic questions refused. |
| V3 | [v3_chroma.py](versions/v3_chroma.py) | … → **ChromaDB** → retriever (top 5) → LLM | A real vector database with metadata: `source` (NIH URL), `page` (section within the answer), `doc_type` (symptoms, treatment, …). |
| V4 | [v4_rerank.py](versions/v4_rerank.py) | ChromaDB (top 20) → **cross-encoder reranker** → top 5 → LLM | Better context: `ms-marco-MiniLM-L-6-v2` reads question and chunk together and keeps the best 5. **Used by the API, React UI and Streamlit app.** |

**FAISS vs ChromaDB.** FAISS is a similarity-search library: vectors are held in memory and saved as files, and metadata sits in a separate pickled docstore. ChromaDB is a vector database: it stores vectors, text and metadata together on disk, supports metadata filters, single-record updates and deletes, and can run as a server. On 6,452 chunks both search in milliseconds and return the same results; Chroma's advantage is managing the data, not speed.

```bash
python3.12 run.py build-versions      # build faiss_index/ and chroma_lc/ (about 1 minute)
python3.12 run.py v1 "How is melanoma treated?"
python3.12 run.py v4 "How do they treat cancer of the voice box?"
python3.12 run.py compare             # V1 vs V2 vs V3 vs V4 on 27 questions (~40 min on the Groq free tier)
python3.12 run.py streamlit           # V4 web app
python3.12 run.py test                # unit tests
```

### Evaluation: V2 vs V3 vs V4

[eval/compare_versions.py](eval/compare_versions.py) runs V2, V3 and V4 (plus V1 as a no-retrieval baseline) on the same [27 questions](data/eval_questions.json): 22 cancer questions written the way users ask them ("cancer of the voice box", "CLL", "my blood is too thick"), each labelled with the NIH answer it should find, plus 5 out-of-scope questions. It measures retrieval quality (hit@5, MRR, precision@5), answer accuracy (an LLM judge grading correct / partial / wrong against the NIH answer, plus BERTScore), refusals and response time. Full results: [docs/version_comparison.md](docs/version_comparison.md).

| Metric | V1 no retrieval | V2 FAISS | V3 ChromaDB | V4 Reranking |
|---|---|---|---|---|
| Retrieval: gold answer in top 5 (hit@5) | – | 96% | 96% | 96% |
| Retrieval: MRR (how high the gold chunk ranks) | – | 0.820 | 0.820 | 0.744 |
| Retrieval: precision@5 | – | 53% | 53% | 56% |
| Answer accuracy (LLM judge) | 82% | 86% | 86% | 84% |
| Judge grades correct / partial / wrong | 14 / 8 / 0 | 17 / 4 / 1 | 17 / 4 / 1 | 15 / 7 / 0 |
| BERTScore F1 vs NIH answer | 0.816 | 0.825 | 0.827 | 0.829 |
| Out-of-scope questions refused | 3 / 5 | 5 / 5 | 5 / 5 | 5 / 5 |
| Sources shown with answer | no | yes | yes | yes |
| Median response time | 1.0 s | 1.29 s | 1.25 s | 1.41 s |

What the numbers say:

- **V1 → V2: grounding is the big step.** Accuracy rises from 82% to 86%. More importantly, every answer now comes with NIH sources and off-topic questions are refused. V1 gave ibuprofen doses and cholesterol advice with no source; V2–V4 refused both.
- **V2 → V3: same quality, better data management.** FAISS and Chroma return the same chunks with identical scores (both use cosine similarity on the same embeddings), so accuracy and retrieval are identical. Chroma adds stored metadata (source, page, document type) and database features. Search time is about 60 ms for both.
- **V3 → V4: the reranker helps on clearly worded questions, not yet on lay wording.** On a separate 40-question development set ([data/dev_questions.json](data/dev_questions.json), different topics), reranking moved the correct chunk to first place almost every time (MRR 0.919 → 0.988). On the 22 test questions, written in heavy lay language ("voice box", "CLL", "womb lining"), the general-purpose MS MARCO cross-encoder sometimes promoted a related cancer (hypopharyngeal for laryngeal, neuroendocrine for pancreatic), lowering MRR to 0.744. Answer quality held steady: V4 is the only version with no wrong answers, and it adds about 60 ms. The first-stage search is already strong (96% hit@5) because each chunk is embedded with its topic and question, which leaves the reranker little headroom.
- **Next improvement:** a medical-domain reranker, or fusing the vector score with the rerank score. The ablation ([eval/rerank_ablation.py](eval/rerank_ablation.py)) compares six reranking set-ups, chosen on the development set and only then checked on the test set.
- With 22 questions, one question equals 4.5 percentage points. Differences of one or two questions between versions are within the variation of the LLM and the judge.

### LangSmith monitoring

1. Create a free account at <https://smith.langchain.com> and an API key (Settings → API Keys).
2. Add to `.env`:
   ```
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=your-key
   LANGSMITH_PROJECT=health-faq-rag
   ```
3. Ask questions in any version, the app or `run.py compare`. Each question appears as a trace in the `health-faq-rag` project: `V4 Rerank RAG` → `retrieve` (retriever with the 20 candidate documents and metadata) → `reranker` (20 in, 5 out with scores) → `generate` → `ChatPromptTemplate` → `ChatGroq` (latency, token usage) → `StrOutputParser`. Failed calls show as error runs. Traces are tagged `v1`–`v4`, `api`, `streamlit` or `eval` for filtering, and `run.py compare` attaches its scores (`judge_accuracy`, `hit_at_5`, `reciprocal_rank`, `correct_refusal`) to each trace as feedback.

Without these variables tracing is off and everything else works the same.

## Deploy to Streamlit Community Cloud

The Streamlit app (`src/app.py`, V4) deploys straight from this GitHub repo. Streamlit Cloud installs [src/requirements.txt](src/requirements.txt) (the file next to the entrypoint), a slim list with CPU-only PyTorch, not the full root `requirements.txt`.

1. Sign in at <https://share.streamlit.io> with GitHub and click **Create app** → **Deploy a public app from GitHub**.
2. Repository `kevincostner17/Health_FAQ_Rag`, branch `main`, main file path `src/app.py`.
3. **Advanced settings** → Python version **3.12**, and paste into **Secrets**:
   ```toml
   GROQ_API_KEY = "your-groq-key"
   # optional monitoring
   LANGSMITH_TRACING = "true"
   LANGSMITH_API_KEY = "your-langsmith-key"
   LANGSMITH_PROJECT = "health-faq-rag"
   ```
4. Click **Deploy**. The first start installs packages and builds the Chroma index from `data/qa_pairs.json` (a few minutes). After that, questions answer in about 2 seconds. The index is rebuilt whenever the app wakes from sleep.

## Course Context

Submitted as part of the WIL Project (Case Studies in Data Science), Milestone 1 — Test-Driven RAG solution with evaluation framework.

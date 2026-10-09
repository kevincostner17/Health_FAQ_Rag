"""FastAPI backend for Health FAQ RAG."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.history import clear_history, get_entry, init_db, list_history, save_entry
from backend.rag_service import ask, get_pipeline

app = FastAPI(title="Health FAQ RAG API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    save_history: bool = True
    compute_metrics: bool = True


class ChatResponse(BaseModel):
    id: int | None
    question: str
    answer: str
    refused: bool
    sources: list
    metrics: dict


@app.on_event("startup")
def startup():
    init_db()
    get_pipeline()  # load V4 models and indexes (built on first run) before the first request


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "health-faq-rag"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = ask(req.question, compute_ragas=req.compute_metrics)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # Groq rate limits / outages: report instead of a bare 500
        raise HTTPException(
            status_code=503, detail=f"Answer service unavailable ({type(e).__name__}). Please try again."
        ) from e

    entry_id = None
    if req.save_history:
        entry_id = save_entry(
            result["question"],
            result["answer"],
            result["refused"],
            result["sources"],
            result["metrics"],
        )

    return ChatResponse(id=entry_id, **result)


@app.get("/api/history")
def history(limit: int = 50):
    return {"items": list_history(limit=min(limit, 200))}


@app.get("/api/history/{entry_id}")
def history_item(entry_id: int):
    item = get_entry(entry_id)
    if not item:
        raise HTTPException(status_code=404, detail="History entry not found")
    return item


@app.delete("/api/history")
def history_clear():
    clear_history()
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)

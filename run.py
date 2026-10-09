#!/usr/bin/env python3
"""
Health FAQ RAG — local runner.

Usage:
  python run.py setup      # install deps + build vector store + test sets
  python run.py pipeline   # run retrieval + generation smoke tests
  python run.py eval       # full Walert-style evaluation
  python run.py ragas      # RAGAS batch metrics (faithfulness, relevance, hallucination)
  python run.py api        # FastAPI backend (port 8000)
  python run.py web        # React frontend (port 5173) — run api in another terminal
  python run.py chat       # terminal chatbot

  Course versions (versions/):
  python run.py build-versions   # build FAISS (V2) and Chroma (V3/V4) indexes
  python run.py v1 ["question"]  # V1 LangChain basics: prompt -> LLM -> parser
  python run.py v2 ["question"]  # V2 FAISS RAG
  python run.py v3 ["question"]  # V3 ChromaDB RAG with metadata
  python run.py v4 ["question"]  # V4 RAG with reranking
  python run.py compare          # evaluate V1 vs V2 vs V3 vs V4 (same 27 questions)
  python run.py streamlit        # Streamlit app on V4 (deployment target)
  python run.py test             # unit tests
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "bin" / "python"


def run(cmd: list[str], cwd: Path | None = None):
    print(f"\n>> {' '.join(cmd)}\n")
    subprocess.check_call(cmd, cwd=cwd or ROOT)


def python() -> str:
    return str(VENV_PY if VENV_PY.exists() else sys.executable)


def cmd_setup():
    if not VENV_PY.exists():
        run([sys.executable, "-m", "venv", str(ROOT / ".venv")])
    run([python(), "-m", "pip", "install", "-r", "requirements.txt"])
    run([python(), "src/03_embed_store.py"])
    run([python(), "eval/build_test_set.py"])
    run([python(), "versions/build_indexes.py"])


def cmd_pipeline():
    run([python(), "src/04_retrieve_test.py"])
    run([python(), "src/05_generate.py"])


def cmd_eval():
    test_sets = ROOT / "data" / "test_sets.json"
    if not test_sets.exists():
        run([python(), "eval/build_test_set.py"])
    run([python(), "src/run_eval.py"])


def cmd_ragas():
    test_sets = ROOT / "data" / "test_sets.json"
    if not test_sets.exists():
        run([python(), "eval/build_test_set.py"])
    run([python(), "eval/ragas_eval.py"] + sys.argv[2:])


def cmd_api():
    uvicorn = ROOT / ".venv" / "bin" / "uvicorn"
    exe = str(uvicorn if uvicorn.exists() else "uvicorn")
    run([exe, "backend.api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])


def cmd_web():
    npm = "npm"
    frontend = ROOT / "frontend"
    node_modules = frontend / "node_modules"
    if not node_modules.exists():
        run([npm, "install"], cwd=frontend)
    run([npm, "run", "dev"], cwd=frontend)


def cmd_chat():
    run([python(), "src/chat.py"])


def cmd_build_versions():
    run([python(), "versions/build_indexes.py"] + sys.argv[2:])


def _version(script: str):
    return lambda: run([python(), f"versions/{script}"] + sys.argv[2:])


def cmd_compare():
    run([python(), "eval/compare_versions.py"] + sys.argv[2:])


def cmd_streamlit():
    run([python(), "-m", "streamlit", "run", "src/app.py"])


def cmd_test():
    run([python(), "-m", "pytest", "tests", "-q"])


COMMANDS = {
    "setup": cmd_setup,
    "pipeline": cmd_pipeline,
    "eval": cmd_eval,
    "ragas": cmd_ragas,
    "api": cmd_api,
    "web": cmd_web,
    "chat": cmd_chat,
    "build-versions": cmd_build_versions,
    "v1": _version("v1_basics.py"),
    "v2": _version("v2_faiss.py"),
    "v3": _version("v3_chroma.py"),
    "v4": _version("v4_rerank.py"),
    "compare": cmd_compare,
    "streamlit": cmd_streamlit,
    "test": cmd_test,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()

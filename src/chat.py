"""
Interactive Health FAQ chatbot (RAG).

Run: python src/chat.py
Requires: chroma_db/ and .env with GROQ_API_KEY (run 03_embed_store.py first).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util

from rag_utils import load_collection


def _load_generate_module():
    path = Path(__file__).parent / "05_generate.py"
    spec = importlib.util.spec_from_file_location("generate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    gen = _load_generate_module()

    try:
        collection = load_collection()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("Health FAQ RAG Chatbot")
    print("Grounded in NIH Cancer.gov (MedQuAD). Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        try:
            answer = gen.answer_question(question, collection)
            print(f"\nBot: {answer}\n")
        except RuntimeError as e:
            print(f"\n[Error] {e}\n")


if __name__ == "__main__":
    main()

"""
Build (or rebuild) the FAISS index for V2 and the Chroma store for V3/V4.

Run: python versions/build_indexes.py [--rebuild]
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from versions.common import CHROMA_DIR, FAISS_DIR
from versions import v2_faiss, v3_chroma


def main():
    if "--rebuild" in sys.argv:
        for path in (FAISS_DIR, CHROMA_DIR):
            if path.exists():
                shutil.rmtree(path)
    v2_faiss.get_vectorstore()
    v3_chroma.get_vectorstore()
    print("Indexes ready.")


if __name__ == "__main__":
    main()

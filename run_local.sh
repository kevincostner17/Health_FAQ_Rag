#!/usr/bin/env bash
# Build and run the Health FAQ RAG pipeline locally (no git push).
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi

if [ ! -f ".env" ]; then
  echo "ERROR: Create .env with GROQ_API_KEY=your-key"
  exit 1
fi

echo "=== Step 3: Embed & store ==="
python src/03_embed_store.py

echo ""
echo "=== Step 4: Retrieval test ==="
python src/04_retrieve_test.py

echo ""
echo "=== Step 5: Generation test ==="
python src/05_generate.py

echo ""
echo "=== Ready. Start chat with: python src/chat.py ==="

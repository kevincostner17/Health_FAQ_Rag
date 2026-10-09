"""
Browser-based Health FAQ RAG demo (Streamlit) running Version 4: Chroma retrieval +
cross-encoder reranking + Groq LLM. Traced in LangSmith when LANGSMITH_TRACING=true.

Run: streamlit run src/app.py
Requires: GROQ_API_KEY in .env (or Streamlit secrets). Indexes are built on first start.
"""
import os
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Hosted platforms (Streamlit Cloud, Hugging Face Spaces) provide keys as secrets, not .env.
try:
    for _key in ("GROQ_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_PROJECT"):
        if _key in st.secrets and _key not in os.environ:
            os.environ[_key] = str(st.secrets[_key])
except FileNotFoundError:
    pass

from versions import v4_rerank


@st.cache_resource(show_spinner="Loading models and search index (first start builds it)...")
def get_rag():
    return v4_rerank.build_chain()


st.set_page_config(page_title="Health FAQ RAG", page_icon="🏥", layout="wide")

# Streamlit has no config to hide only "Record a screencast" — hide it via JS, keep Print/Rerun/etc.
components.html(
    """
    <script>
    (function () {
        const HIDE = /record\\s*a\\s*screencast/i;
        function hideRecordOption() {
            const doc = window.parent.document;
            doc.querySelectorAll('[role="menuitem"], li, button, a, span, p, div').forEach(function (el) {
                const text = (el.textContent || "").trim();
                if (HIDE.test(text)) {
                    const row = el.closest("li") || el.closest('[role="menuitem"]') || el;
                    row.style.display = "none";
                }
            });
        }
        hideRecordOption();
        new MutationObserver(hideRecordOption).observe(window.parent.document.body, { childList: true, subtree: true });
    })();
    </script>
    """,
    height=0,
)

st.title("Health FAQ RAG Chatbot")
st.caption("Grounded in NIH Cancer.gov (MedQuAD) · V4: Chroma retrieval + cross-encoder reranking")

try:
    get_rag()
except Exception as e:
    st.error(f"Failed to load the RAG system: {e}")
    st.stop()

with st.sidebar:
    st.header("About")
    st.markdown(
        "Ask consumer health questions about cancer. "
        "Answers are generated only from retrieved NIH sources. "
        "Out-of-scope questions should be refused."
    )
    st.divider()
    show_sources = st.checkbox("Show retrieved sources", value=True)
    st.divider()
    st.markdown("**Example questions**")
    examples = [
        "What are the symptoms of leukemia?",
        "What are the treatments for Prostate Cancer ?",
        "What causes lung cancer?",
        "What is the capital of France?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["question"] = ex

question = st.text_input(
    "Your question",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What are the symptoms of breast cancer?",
)

if st.button("Ask", type="primary") and question.strip():
    if len(question) > 500:
        st.error("Please keep your question under 500 characters.")
        st.stop()
    with st.spinner("Retrieving, reranking and generating answer..."):
        try:
            result = v4_rerank.ask(question, config={"tags": ["streamlit"]})
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"The answer service is unavailable right now ({type(e).__name__}). Please try again.")
            st.stop()

    st.subheader("Answer")
    st.markdown(result["answer"])
    tm = result["timings"]
    st.caption(
        f"Retrieval {tm['retrieve_ms']:.0f} ms · rerank {tm.get('rerank_ms', 0):.0f} ms · "
        f"LLM {tm['generate_ms']:.0f} ms · total {tm['total_ms']:.0f} ms"
    )

    if show_sources:
        st.subheader("Source documents")
        if result["refused"]:
            st.info("No relevant NIH passages found, so the question was treated as out of scope.")
        for i, s in enumerate(result["sources"], 1):
            label = f"{i}. {s['focus']} · {s['doc_type']} · page {s['page']}/{s['total_pages']}"
            with st.expander(f"{label} (rerank {s['rerank_score']:.2f}, similarity {s['similarity']:.2f})"):
                st.markdown(f"**Source:** [{s['url']}]({s['url']})")
                st.markdown(f"**Matched NIH question:** {s['question']}")
                st.text(s["chunk"])

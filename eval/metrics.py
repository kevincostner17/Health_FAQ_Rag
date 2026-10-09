"""Evaluation metrics mirroring Walert's Known / Inferred / Out-of-KB categories."""

REFUSAL_PHRASES = (
    "don't have enough information",
    "do not have enough information",
    "not enough information in my health knowledge base",
    "cannot answer that question",
    "can't answer that question",
    "provided information does not include",
    "provided excerpts do not include",
    "does not include any details about",
    "does not include information about",
)


def looks_like_refusal(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in REFUSAL_PHRASES)


def rouge1_score(prediction: str, reference: str) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores["rouge1"].fmeasure


def bertscore_f1(prediction: str, reference: str) -> float:
    from bert_score import score as bert_score

    _, _, f1 = bert_score([prediction], [reference], lang="en", verbose=False)
    return float(f1[0])

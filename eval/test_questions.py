"""
Evaluation test question sets, structured to mirror Walert's evaluation
categories: Known, Inferred, and Out-of-KB.
"""

KNOWN_QUESTIONS = [
    {"question": "What are the symptoms of leukemia?", "gold_answer": None},
    {"question": "What are the symptoms of Adult Acute Lymphoblastic Leukemia ?", "gold_answer": None},
    {"question": "How is breast cancer treated?", "gold_answer": None},
    {"question": "What are the treatments for Prostate Cancer ?", "gold_answer": None},
    {"question": "What causes lung cancer?", "gold_answer": None},
    {"question": "Who is at risk for Adult Acute Lymphoblastic Leukemia?", "gold_answer": None},
]

INFERRED_QUESTIONS = [
    {"question": "What blood-related problems can leukemia cause?"},
]

OUT_OF_KB_QUESTIONS = [
    {"question": "What is the capital of France?"},
    {"question": "What's the best diet for training a marathon?"},
    {"question": "How do I fix a flat tire on my bicycle?"},
]

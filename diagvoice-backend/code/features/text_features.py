"""Transcript üzerinden basit skaler metin özellikleri."""
from __future__ import annotations

import math
import re
from typing import Any


def extract_text_features_dict(transcript: str) -> dict[str, Any]:
    t = transcript or ""
    words = re.findall(r"\S+", t)
    n_words = len(words)
    n_chars = len(t)
    n_lines = t.count("\n") + 1 if t else 0
    upper = sum(1 for c in t if c.isupper())
    digit = sum(1 for c in t if c.isdigit())

    # Eğitim (04_train_eval.scalar_columns): text_* — txt_* sütunları skaler modele girmez
    return {
        "text_char_len": float(n_chars),
        "text_word_count": float(n_words),
        "text_line_count": float(n_lines),
        "text_avg_word_len": float(n_chars / max(n_words, 1)),
        "text_upper_ratio": float(upper / max(n_chars, 1)),
        "text_digit_ratio": float(digit / max(n_chars, 1)),
        "text_exclaim_count": float(t.count("!")),
        "text_question_count": float(t.count("?")),
    }

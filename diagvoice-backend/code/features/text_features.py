"""Transkript üzerinden skalar metin özellikleri (Türkçe)."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from .lexicon_tr import (
    FILLERS_TR,
    FIRST_PERSON_PRONOUNS_TR,
    FUTURE_SUFFIX_HINTS,
    NEGATIVE_WORDS_TR,
    PAST_SUFFIX_HINTS,
    POSITIVE_WORDS_TR,
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENT_SPLIT = re.compile(r"[.!?…]+")
_VOWELS_TR = frozenset("aeıioöuüAEIİIOÖUÜ")

TEXT_SCALAR_KEYS: tuple[str, ...] = (
    "text_word_count",
    "text_unique_word_count",
    "text_type_token_ratio",
    "text_sentence_count",
    "text_avg_sentence_length_words",
    "text_question_sentence_ratio",
    "text_incomplete_sentence_ratio_heuristic",
    "text_pronoun_first_person_ratio",
    "text_negative_word_ratio",
    "text_positive_word_ratio",
    "text_sentiment_neg_minus_pos_ratio",
    "text_filler_ratio",
    "text_past_marker_ratio_heuristic",
    "text_future_marker_ratio_heuristic",
    "text_word_repetition_ratio",
    "text_syllable_count_estimate",
)


def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def tokenize(text: str) -> list[str]:
    return [tr_lower(m.group(0)) for m in _WORD_RE.finditer(text or "")]


def split_sentences(text: str) -> list[str]:
    if not (text or "").strip():
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def count_syllables_tr_word(word: str) -> int:
    w = tr_lower(word)
    if not w:
        return 0
    n = 0
    prev_v = False
    for ch in w:
        v = ch in _VOWELS_TR
        if v and not prev_v:
            n += 1
        prev_v = v
    return max(n, 1)


def count_syllables_tr_text(tokens: list[str]) -> int:
    return sum(count_syllables_tr_word(t) for t in tokens)


def _token_has_past_hint(tok: str) -> bool:
    return any(tok.endswith(s) for s in PAST_SUFFIX_HINTS)


def _token_has_future_hint(tok: str) -> bool:
    return any(s in tok for s in FUTURE_SUFFIX_HINTS)


def _sentence_lacks_finite_verb_heuristic(sent: str, tokens: list[str]) -> bool:
    if len(tokens) <= 2:
        return True
    if len(tokens) >= 12:
        joined = " ".join(tokens)
        if not re.search(
            r"(yor|dı|di|du|dü|miş|mus|müş|mış|ecek|acak|malı|abil|ır|ir|ur|ür|ar|er)\b",
            joined,
        ):
            return True
    return False


def extract_text_features_dict(text: str) -> dict[str, Any]:
    text = text or ""
    tokens = tokenize(text)
    sentences = split_sentences(text)
    sent_token_lists = [tokenize(s) for s in sentences] if sentences else [tokens]

    wc = len(tokens)
    uq = len(set(tokens)) if tokens else 0
    sc = len(sentences) if sentences else (1 if wc else 0)

    out: dict[str, Any] = {
        "text_word_count": float(wc),
        "text_unique_word_count": float(uq),
        "text_type_token_ratio": (uq / wc) if wc else math.nan,
        "text_sentence_count": float(sc),
        "text_avg_sentence_length_words": (wc / sc) if sc else math.nan,
    }

    qn = sum(1 for s in sentences if "?" in s) if sentences else 0
    out["text_question_sentence_ratio"] = (qn / sc) if sc else math.nan

    incomplete = 0
    for s, stoks in zip(sentences, sent_token_lists):
        if _sentence_lacks_finite_verb_heuristic(s, stoks):
            incomplete += 1
    out["text_incomplete_sentence_ratio_heuristic"] = (incomplete / sc) if sc else math.nan

    if wc:
        pron = sum(1 for t in tokens if t in FIRST_PERSON_PRONOUNS_TR)
        neg = sum(1 for t in tokens if t in NEGATIVE_WORDS_TR)
        pos = sum(1 for t in tokens if t in POSITIVE_WORDS_TR)
        fill = sum(1 for t in tokens if t in FILLERS_TR)
        past = sum(1 for t in tokens if _token_has_past_hint(t))
        fut = sum(1 for t in tokens if _token_has_future_hint(t))
        out["text_pronoun_first_person_ratio"] = pron / wc
        out["text_negative_word_ratio"] = neg / wc
        out["text_positive_word_ratio"] = pos / wc
        out["text_sentiment_neg_minus_pos_ratio"] = (neg - pos) / wc
        out["text_filler_ratio"] = fill / wc
        out["text_past_marker_ratio_heuristic"] = past / wc
        out["text_future_marker_ratio_heuristic"] = fut / wc
        out["text_word_repetition_ratio"] = 1.0 - (uq / wc)
    else:
        for k in (
            "text_pronoun_first_person_ratio",
            "text_negative_word_ratio",
            "text_positive_word_ratio",
            "text_sentiment_neg_minus_pos_ratio",
            "text_filler_ratio",
            "text_past_marker_ratio_heuristic",
            "text_future_marker_ratio_heuristic",
            "text_word_repetition_ratio",
        ):
            out[k] = math.nan

    out["text_syllable_count_estimate"] = float(count_syllables_tr_text(tokens))
    return out


def text_scalar_matrix(texts: list[str]) -> np.ndarray:
    """Sabit sırada skalar metin özellikleri matrisi (n, len(TEXT_SCALAR_KEYS))."""
    rows: list[list[float]] = []
    for t in texts:
        d = extract_text_features_dict(t)
        rows.append([float(d.get(k, math.nan)) for k in TEXT_SCALAR_KEYS])
    return np.asarray(rows, dtype=np.float64)

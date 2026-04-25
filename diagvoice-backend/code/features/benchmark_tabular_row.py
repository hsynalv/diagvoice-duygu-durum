"""
benchmark_v2 tabular adimi ile ayni sozluk (ses + metin skalar + eGeMAPS
[+ istege bagli metin gomme]). Bu kopya inference icindir.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Any

import numpy as np

from .audio_features import extract_audio_features_dict
from .opensmile_features import extract_opensmile_egemaps_dict
from .text_features import extract_text_features_dict


def extract_tabular_features_for_benchmark(
    audio_path: str,
    transcript: str,
    *,
    target_sr: int = 16000,
    top_db: float = 35.0,
    hop_length: int = 512,
    run_praat_voice: bool = False,
    use_opensmile_egemaps: bool = False,
    strict_opensmile: bool = False,
) -> dict[str, Any]:
    osm_fn = None
    if use_opensmile_egemaps:
        try:
            import opensmile  # noqa: F401

            osm_fn = extract_opensmile_egemaps_dict
        except ImportError as e:
            if strict_opensmile:
                raise RuntimeError(
                    "config'te openSMILE acik (opensmile_egemaps) ama 'opensmile' paketi yok. "
                    "pip install opensmile"
                ) from e
            print(
                f"[uyari] openSMILE (eGeMAPS) atlaniyor: {e} - pip install opensmile",
                file=sys.stderr,
            )

    audio_d = extract_audio_features_dict(
        audio_path,
        target_sr=target_sr,
        top_db=top_db,
        hop_length=hop_length,
        run_praat_voice=run_praat_voice,
    )
    text_d = extract_text_features_dict(transcript)

    row: dict[str, Any] = {**audio_d, **text_d}
    if osm_fn is not None:
        try:
            row.update(osm_fn(str(audio_path), target_sr=target_sr))
        except Exception as e:
            if strict_opensmile:
                raise
            print(f"[uyari-opensmile] {audio_path}: {e}", file=sys.stderr)
    return row


@lru_cache(maxsize=4)
def _get_sentence_transformer(model_id: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


def add_embed_doc_columns(
    row: dict[str, Any],
    text: str,
    *,
    model_id: str,
    batch_size: int = 16,
) -> None:
    st = _get_sentence_transformer(model_id)
    t = text or " "
    vecs = st.encode(
        [t],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    v = np.asarray(vecs[0], dtype=np.float64).ravel()
    for k, val in enumerate(v):
        row[f"embed_doc_{k}"] = float(val)


def build_tabular_dict_from_config(
    audio_path: str,
    transcript: str,
    cfg: dict[str, Any],
    *,
    with_text_embeddings: bool | None = None,
    strict_opensmile: bool = False,
) -> dict[str, Any]:
    af = cfg.get("audio_features") or {}
    te = cfg.get("text_embeddings") or {}
    if with_text_embeddings is None:
        with_text_embeddings = bool(te.get("enabled"))

    row = extract_tabular_features_for_benchmark(
        audio_path,
        transcript,
        target_sr=int(af.get("target_sr", 16000)),
        top_db=float(af.get("top_db", 35.0)),
        hop_length=int(af.get("hop_length", 512)),
        run_praat_voice=bool(af.get("praat_voice", False)),
        use_opensmile_egemaps=bool(af.get("opensmile_egemaps", False)),
        strict_opensmile=strict_opensmile,
    )
    if with_text_embeddings and te.get("model"):
        add_embed_doc_columns(
            row,
            transcript,
            model_id=str(te.get("model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")),
            batch_size=int(te.get("batch_size", 16)),
        )
    return row

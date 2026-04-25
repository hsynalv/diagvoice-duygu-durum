"""
Ses dosyasından librosa tabanlı skaler özellikler.
Sütun adları `training/04_train_eval.scalar_columns` ile uyumlu: `audio_*` öneki (osm_* ayrı modül).
Praat / farklı istatistik seti eğitimde kullanıldıysa DIAGVOICE_CODE_ROOT ile tam pipeline kopyalayın.
"""
from __future__ import annotations

import math
from typing import Any

import librosa
import numpy as np


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def extract_audio_features_dict(
    audio_path: str,
    target_sr: int = 16000,
    top_db: float = 35.0,
    hop_length: int = 512,
    run_praat_voice: bool = False,
) -> dict[str, Any]:
    del run_praat_voice  # Docker minimal imageda Praat yok; yoksay

    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    if y.size == 0:
        return {"audio_duration_sec": 0.0}

    duration = float(len(y) / sr)
    y_trim, _ = librosa.effects.trim(y, top_db=top_db)

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    try:
        tempo_arr = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
    except AttributeError:
        tempo_arr = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    tempo = float(np.asarray(tempo_arr).ravel()[0]) if tempo_arr.size else float("nan")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)

    def stat(prefix: str, arr: np.ndarray) -> dict[str, float]:
        a = np.asarray(arr, dtype=np.float64).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {f"{prefix}_mean": float("nan"), f"{prefix}_std": float("nan")}
        return {f"{prefix}_mean": _safe_float(np.mean(a)), f"{prefix}_std": _safe_float(np.std(a))}

    # Eğitim (04_train_eval.scalar_columns): yalnızca audio_ / text_ / meta_ / osm_ önekleri
    out: dict[str, Any] = {"audio_duration_sec": duration}
    out.update(stat("audio_rms", rms))
    out.update(stat("audio_zcr", zcr))
    out.update(stat("audio_spec_cent", cent))
    out.update(stat("audio_spec_roll", rolloff))
    out.update(stat("audio_spec_bw", bandwidth))
    out.update(stat("audio_spec_flat", flatness))
    out["audio_tempo_bpm"] = tempo

    for i in range(min(13, mfcc.shape[0])):
        out.update(stat(f"audio_mfcc_{i}", mfcc[i]))
        out.update(stat(f"audio_mfcc_d_{i}", delta[i]))

    for k in range(min(7, contrast.shape[0])):
        out.update(stat(f"audio_spec_contrast_{k}", contrast[k]))

    rms_trim = librosa.feature.rms(y=y_trim, hop_length=hop_length)[0] if y_trim.size else rms
    out["audio_rms_trim_mean"] = _safe_float(np.mean(rms_trim))

    return out

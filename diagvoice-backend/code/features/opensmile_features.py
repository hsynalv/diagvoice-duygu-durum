"""openSMILE eGeMAPS v2 functionals (audeering opensmile-python)."""

from __future__ import annotations

import os
import re
import tempfile

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

try:
    import soundfile as sf
except ImportError:
    sf = None


def _sanitize_key(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "f"


def extract_opensmile_egemaps_dict(
    audio_path: str,
    *,
    target_sr: int = 16000,
) -> dict[str, float]:
    """eGeMAPS v02 functionals -> osm_* skalar sozlugu. WAV disi formatlar gecici WAV ile okunur."""
    import opensmile  # type: ignore  # noqa: PLC0415

    if librosa is None or sf is None:
        raise ImportError("librosa ve soundfile gerekli (opensmile on isleme icin)")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    p = str(audio_path)
    lower = p.lower()
    tmp: str | None = None
    try:
        if lower.endswith(".wav"):
            proc = p
        else:
            y, sr = librosa.load(p, sr=target_sr, mono=True)
            y = np.asarray(y, dtype=np.float32)
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(tmp, y, int(sr))
            proc = tmp
        df = smile.process_file(proc)
    finally:
        if tmp and os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    if df is None or len(df) < 1:
        return {}
    row = df.iloc[0]
    out: dict[str, float] = {}
    for k, v in row.items():
        key = _sanitize_key(k)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fv):
            continue
        out[f"osm_{key}"] = fv
    return out

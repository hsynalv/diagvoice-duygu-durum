"""openSMILE eGeMAPS — opsiyonel; paket yoksa boş dict."""
from __future__ import annotations

from typing import Any


def extract_opensmile_egemaps_dict(audio_path: str, target_sr: int = 16000) -> dict[str, Any]:
    try:
        import opensmile  # type: ignore

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        y = smile.process_file(audio_path)
        out: dict[str, Any] = {}
        for c in y.columns:
            v = float(y[c].iloc[0])
            out[f"osm_{c}"] = v
        return out
    except Exception:
        return {}

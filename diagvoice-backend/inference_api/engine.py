"""
Benchmark (is_sağlığı / benchmark_v2) joblib inference motoru — Flask'tan bağımsız.
Fusion ve inference_api.app bu modülü kullanır.
"""
from __future__ import annotations

import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DEFAULT_MODEL = MODELS_DIR / "model.joblib"
DEFAULT_CONFIG = APP_DIR / "config.yaml"

ROOT = APP_DIR.parent
_CODE = ROOT / "code"

_bundle: dict[str, Any] | None = None
_model_path: Path | None = None
_st_model = None
_st_model_id: str | None = None
_paths_setup = False


def setup_sys_path() -> None:
    """Özellik modülleri: önce DIAGVOICE_CODE_ROOT, sonra /app/code (Docker)."""
    global _paths_setup
    if _paths_setup:
        return
    extra = os.environ.get("DIAGVOICE_CODE_ROOT", "").strip()
    if extra:
        p = Path(extra).expanduser().resolve()
        if p.is_dir():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            code_alt = p / "code"
            if code_alt.is_dir() and str(code_alt) not in sys.path:
                sys.path.insert(0, str(code_alt))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if _CODE.is_dir() and str(_CODE) not in sys.path:
        sys.path.insert(0, str(_CODE))
    _paths_setup = True


def _load_yaml(p: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML gerekli: pip install pyyaml") from e
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_config_path() -> Path:
    return Path(os.environ.get("DIAGVOICE_BENCHMARK_CONFIG", str(DEFAULT_CONFIG))).resolve()


def _cfg_audio() -> dict[str, Any]:
    cfg = _load_yaml(load_config_path())
    return cfg.get("audio_features") or {}


def resolve_model_path() -> Path:
    env = os.environ.get("DIAGVOICE_MODEL")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"DIAGVOICE_MODEL bulunamadı: {p}")
        return p
    if DEFAULT_MODEL.is_file():
        return DEFAULT_MODEL
    raise FileNotFoundError(
        f"Varsayılan model yok: {DEFAULT_MODEL}. "
        f"model.joblib koyun veya DIAGVOICE_MODEL ile yol verin."
    )


def load_bundle() -> tuple[Path, dict[str, Any]]:
    global _bundle, _model_path
    if _bundle is not None and _model_path is not None:
        return _model_path, _bundle
    setup_sys_path()
    _model_path = resolve_model_path()
    _bundle = joblib.load(_model_path)
    if not isinstance(_bundle, dict):
        raise TypeError("joblib yükü geçerli bir sözlük değil")
    return _model_path, _bundle


def reset_bundle_cache() -> None:
    global _bundle, _model_path
    _bundle = None
    _model_path = None


def _get_sentence_transformer(model_id: str):
    global _st_model, _st_model_id
    if _st_model is not None and _st_model_id == model_id:
        return _st_model
    from sentence_transformers import SentenceTransformer

    _st_model = SentenceTransformer(model_id)
    _st_model_id = model_id
    return _st_model


def _vectorize(cols: list[str], row: dict[str, Any]) -> np.ndarray:
    r = np.array([[float("nan") if c not in row or row[c] is None else row[c] for c in cols]], dtype=np.float64)
    return r


def _row_from_audio(audio_path: str, transcript: str, af: dict[str, Any]) -> dict[str, Any]:
    setup_sys_path()
    from features.audio_features import extract_audio_features_dict
    from features.text_features import extract_text_features_dict

    out: dict[str, Any] = {}
    t_sr = int(af.get("target_sr", 16000))
    top_db = float(af.get("top_db", 35.0))
    hop = int(af.get("hop_length", 512))
    praat = bool(af.get("praat_voice", False))
    use_osm = bool(af.get("opensmile_egemaps", False))

    a = extract_audio_features_dict(
        audio_path,
        target_sr=t_sr,
        top_db=top_db,
        hop_length=hop,
        run_praat_voice=praat,
    )
    for k, v in a.items():
        out[k] = float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else v

    td = extract_text_features_dict(transcript)
    for k, v in td.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out[k] = float("nan")
        else:
            out[k] = float(v)

    if use_osm:
        try:
            from features.opensmile_features import extract_opensmile_egemaps_dict

            o = extract_opensmile_egemaps_dict(audio_path, target_sr=t_sr)
            for k, v in o.items():
                out[k] = float(v)
        except Exception:
            pass
    return out


def _fill_text_embeddings(
    row: dict[str, Any], emb_cols: list[str], text: str, model_id: str, batch_size: int = 16
) -> None:
    st = _get_sentence_transformer(model_id)
    vec = st.encode(
        [text or " "],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    v = np.asarray(vec, dtype=np.float64).ravel()
    for c in emb_cols:
        m = re.match(r"^embed_doc_(\d+)$", str(c))
        if m:
            j = int(m.group(1))
            row[c] = float(v[j]) if j < len(v) else float("nan")
        else:
            row[c] = float("nan")


def predict_proba_from_bundle(
    bundle: dict[str, Any], row: dict[str, Any], config: dict[str, Any]
) -> float:
    mode = str(bundle.get("mode", "unknown"))

    if mode == "ssl_only":
        raise NotImplementedError(
            "ssl_only: wav2vec ssl_dim_* bu uç noktada üretilmiyor; scalar_hgb veya scalar_only kullanın."
        )

    if mode in ("scalar_only", "scalar_hgb"):
        cols = list(bundle.get("feature_columns") or [])
        if not cols:
            raise ValueError("bundle'da feature_columns yok")

        X = _vectorize(cols, row)
        t = bundle["imputer"].transform(X)
        t = bundle["scaler"].transform(t)
        pca = bundle.get("block_pca")
        if pca is not None:
            t = pca.transform(t)
        clf = bundle["classifier"]
        return float(clf.predict_proba(t)[0, 1])

    if mode == "scalar_embed_pca":
        sc = list(bundle.get("scalar_columns") or [])
        emb = list(bundle.get("embed_columns") or [])
        if not sc or not emb:
            raise ValueError("scalar_embed_pca: scalar_columns / embed_columns eksik")
        te = config.get("text_embeddings") or {}
        model_id = str(te.get("model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
        _fill_text_embeddings(
            row,
            emb,
            str(config.get("transcript", "") or ""),
            model_id,
            int(te.get("batch_size", 16)),
        )
        Xs = _vectorize(sc, row)
        Xe = _vectorize(emb, row)
        Xs = bundle["scalar_imputer"].transform(Xs)
        Xs = bundle["scalar_scaler"].transform(Xs)
        t = bundle["embed_imputer"].transform(Xe)
        t = bundle["embed_scaler"].transform(t)
        t = bundle["embed_pca"].transform(t)
        h = np.hstack([Xs, t])
        return float(bundle["classifier"].predict_proba(h)[0, 1])

    raise NotImplementedError(
        f"Mod '{mode}' desteklenmiyor. scalar_only, scalar_hgb veya scalar_embed_pca .joblib kullanın."
    )


def threshold_from_bundle(bundle: dict[str, Any]) -> tuple[float, str]:
    t = float(bundle.get("threshold_tuned", 0.5))
    obj = str(bundle.get("threshold_objective", "balanced_accuracy"))
    return t, obj


def predict_from_uploaded_file(tmp_path: str, transcript: str, _ext: str = "") -> dict[str, Any]:
    """Diskteki ses yolu + transcript → Flask /predict ile aynı JSON gövdesi."""
    setup_sys_path()
    cfg_full = _load_yaml(load_config_path())
    cfg_full = dict(cfg_full)
    cfg_full["transcript"] = transcript.strip()
    af = _cfg_audio()

    _, bundle = load_bundle()
    t_thresh, t_obj = threshold_from_bundle(bundle)

    row = _row_from_audio(tmp_path, transcript.strip(), af)
    p_pos = predict_proba_from_bundle(bundle, row, cfg_full)
    y_hat = 1 if p_pos >= t_thresh else 0

    return {
        "mode": bundle.get("mode"),
        "positive_class_probability": p_pos,
        "threshold_tuned": t_thresh,
        "threshold_objective": t_obj,
        "predicted_class": y_hat,
        "class_names_hint": "0: sağlıklı, 1: tanılı (eğitim etiketine bağlı — benchmark ile doğrulayın)",
    }


def health_info() -> dict[str, Any]:
    path, b = load_bundle()
    return {
        "ok": True,
        "model_path": str(path),
        "mode": b.get("mode", "?"),
        "config": str(load_config_path()),
    }

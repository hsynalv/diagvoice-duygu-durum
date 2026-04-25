"""Eğitim betikleri: veri yolları benchmark hattındaki `experiments/benchmark_v2` ile aynıdır.

Düzen A — bu repo: `diagvoice-duygu-durum/training/` + `diagvoice-backend/inference_api/`.
Düzen B — tek kök: `training/` doğrudan `inference_api/` altında (INFERENCE_API.parent = backend).

04_train_eval, API ile aynı `config.yaml` ve `models/model.joblib` hedefini kullanır;
merged/splits parquet’ler 03 adımıyla üretilir.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

TRAINING_DIR = Path(__file__).resolve().parent
_cand = TRAINING_DIR.parent
_backend = _cand / "diagvoice-backend"
if (_backend / "inference_api").is_dir():
    PROJECT_ROOT = _backend.resolve()
    INFERENCE_API = PROJECT_ROOT / "inference_api"
elif (_cand / "inference_api").is_dir():
    INFERENCE_API = (_cand / "inference_api").resolve()
    PROJECT_ROOT = INFERENCE_API.parent
else:
    PROJECT_ROOT = _cand.resolve()
    INFERENCE_API = PROJECT_ROOT / "inference_api"

# Eğitim çıktıları: benchmark kökü (repo kökünde experiments/ veya backend altında)
BENCHMARK_ROOT = _cand / "experiments" / "benchmark_v2"
if not BENCHMARK_ROOT.is_dir() and (PROJECT_ROOT / "experiments" / "benchmark_v2").is_dir():
    BENCHMARK_ROOT = PROJECT_ROOT / "experiments" / "benchmark_v2"

CODE_DIR = PROJECT_ROOT / "code"
SSL_AUDIO_DIR = _cand / "experiments" / "ssl_audio"
if not SSL_AUDIO_DIR.is_dir() and (PROJECT_ROOT / "experiments" / "ssl_audio").is_dir():
    SSL_AUDIO_DIR = PROJECT_ROOT / "experiments" / "ssl_audio"

OUTPUTS = BENCHMARK_ROOT / "outputs"
ARTIFACTS_DIR = OUTPUTS / "artifacts"
SPLITS_DIR = OUTPUTS / "splits"
FEATURES_DIR = OUTPUTS / "features"
REPORTS_DIR = OUTPUTS / "reports"
MODELS_DIR = OUTPUTS / "models"

# API’de yüklenen tekil ağırlık
API_MODEL_JOBLIB = INFERENCE_API / "models" / "model.joblib"
DEFAULT_BENCHMARK_CONFIG = INFERENCE_API / "config.yaml"

HEALTHY_FOLDER = "sağlıklı"
DIAGNOSED_FOLDER = "tanılı tümü"


def add_code_to_syspath() -> None:
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict(data) if isinstance(data, dict) else {}


def ensure_benchmark_output_dirs() -> None:
    for d in (ARTIFACTS_DIR, SPLITS_DIR, FEATURES_DIR, REPORTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def resolve_under_root(path_str: str | Path, root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()

#!/usr/bin/env python3
"""
Holdout değerlendirme: train → (iç val ile C ve eşik) → test.
Modlar: scalar_only, scalar_embed_pca, ssl_only, early_concat_all, late_fusion_logreg.

İyileştirmeler (config eval):
  - threshold_objective: iç val eşiği — balanced_accuracy (varsayılan), f1_positive, youden_j
  - ssl_pca_components: SSL bloğuna train-uyumlu PCA
  - late_fusion: OOF + meta girdü StandardScaler; late_fusion_meta_C ile L2 gücü

Metrikler: roc_auc, pr_auc, F1, balanced_accuracy, sensitivity/specificity (eşik ayarlı), confusion_matrix.
Kök alanlardaki metrikler test kümesi içindir. Raporlara ayrıca aynı (val’da seçilen) eşikle train ve val: {...} eklenir.

Modlar (özet): scalar_only, scalar_hgb, scalar_embed_pca, ssl_only, early_concat_all, early_concat_hgb,
late_fusion_logreg, late_fusion_logreg3 (metin gömme gerekli).

Bu repo: `training/04_train_eval.py` — API ile aynı `diagvoice-backend/inference_api/config.yaml` ve
`.../models/model.joblib` hedefi. Veri: `experiments/benchmark_v2/outputs` (merged.parquet, split_assignments).

Örnek (repo kökünden, scalar_hgb + API model kopyası):
  python training/04_train_eval.py --modes scalar_hgb --save-models --copy-to-api-model
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from _common import (
    API_MODEL_JOBLIB,
    DEFAULT_BENCHMARK_CONFIG,
    FEATURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
    ensure_benchmark_output_dirs,
    load_yaml,
)

DEFAULT_CONFIG = DEFAULT_BENCHMARK_CONFIG
DEFAULT_MERGED = FEATURES_DIR / "merged.parquet"
DEFAULT_SPLITS = SPLITS_DIR / "split_assignments.parquet"

META_SKIP = {
    "source_relpath",
    "class_folder",
    "split",
    "audio_path",
    "transcript_path",
    "participant_stem",
    "participant_core_stem",
    "recording_modality",
    "error",
    "label",
    "ssl_model",
    "ssl_pooling",
}


def _numeric_df_columns(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        if c in META_SKIP:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def scalar_columns(df: pd.DataFrame) -> list[str]:
    cols = _numeric_df_columns(df)
    return [
        c
        for c in cols
        if (
            c.startswith("audio_")
            or c.startswith("text_")
            or c.startswith("meta_")
            or c.startswith("osm_")
        )
        and not c.startswith("embed_doc_")
    ]


def ssl_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in _numeric_df_columns(df) if c.startswith("ssl_dim_")]


def embed_doc_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("embed_doc_") and pd.api.types.is_numeric_dtype(df[c])]


def columns_observed_in_train(df: pd.DataFrame, train_idx: pd.Index, cols: list[str]) -> list[str]:
    """Train bölmesinde en az bir gözlemi olan sütunlar (tamamı NaN olanları çıkar).

    Praat kapalıyken tüm örneklerde NaN kalan akustik alanlar veya eksik SSL boyutları
    SimpleImputer uyarısı üretmesin diye burada elenir; seçim yalnızca train'e bakar (test'e bakılmaz).
    """
    if not cols:
        return []
    sub = df.loc[train_idx, cols]
    return [c for c in cols if sub[c].notna().any()]


def best_threshold_f1(y_true: np.ndarray, proba_pos: np.ndarray) -> tuple[float, float]:
    ts = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        y_pred = (proba_pos >= t).astype(np.int32)
        f = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


def best_threshold_balanced_accuracy(y_true: np.ndarray, proba_pos: np.ndarray) -> tuple[float, float]:
    ts = np.linspace(0.05, 0.95, 19)
    best_t, best_ba = 0.5, -1.0
    for t in ts:
        y_pred = (proba_pos >= t).astype(np.int32)
        s = float(balanced_accuracy_score(y_true, y_pred))
        if s > best_ba:
            best_ba, best_t = s, float(t)
    return best_t, best_ba


def best_threshold_youden_j(y_true: np.ndarray, proba_pos: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(np.int32)
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    fpr, tpr, thresholds = roc_curve(y_true, proba_pos)
    j = tpr - fpr
    idx = int(np.argmax(j))
    raw_t = thresholds[idx]
    t = float(np.clip(raw_t, 0.0, 1.0)) if np.isfinite(raw_t) else 0.5
    return t, float(j[idx])


def pick_threshold(y_true: np.ndarray, proba_pos: np.ndarray, objective: str) -> tuple[float, float]:
    """(eşik, iç val skoru). objective: f1_positive | balanced_accuracy | youden_j"""
    o = objective.strip().lower()
    if o == "f1_positive":
        return best_threshold_f1(y_true, proba_pos)
    if o in ("balanced_accuracy", "balanced_acc", "ba"):
        return best_threshold_balanced_accuracy(y_true, proba_pos)
    if o in ("youden_j", "youden", "j"):
        return best_threshold_youden_j(y_true, proba_pos)
    raise ValueError(f"Bilinmeyen threshold_objective: {objective}")


def sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    cm = confusion_matrix(y_true, y_pred)
    if cm.size != 4:
        return 0.0, 0.0
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return sens, spec


def transform_blocks(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    tr = sc.fit_transform(imp.fit_transform(train))
    va = sc.transform(imp.transform(val))
    te = sc.transform(imp.transform(test))
    return tr, va, te, imp, sc


def pca_fit_transform_train_only(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    max_components: int,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, PCA]:
    n_comp = min(max_components, train.shape[1], max(1, train.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=random_state)
    tr = pca.fit_transform(train)
    va = pca.transform(val)
    te = pca.transform(test)
    return tr, va, te, int(n_comp), pca


def oof_branch_positive_proba(
    X_raw: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int,
    C_fixed: float,
    random_state: int,
) -> np.ndarray:
    """Train seti içinde StratifiedKFold ile OOF pozitif sınıf olasılığı (meta eğitimi için)."""
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for tr_pos, ho_pos in skf.split(np.zeros(n), y):
        X_fit = X_raw[tr_pos]
        y_fit = y[tr_pos]
        X_ho = X_raw[ho_pos]
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        Xf = sc.fit_transform(imp.fit_transform(X_fit))
        Xh = sc.transform(imp.transform(X_ho))
        lr = LogisticRegression(
            C=C_fixed,
            class_weight="balanced",
            max_iter=8000,
            solver="lbfgs",
            random_state=random_state,
        )
        lr.fit(Xf, y_fit)
        oof[ho_pos] = lr.predict_proba(Xh)[:, 1]
    return oof


def oof_text_embed_branch_proba(
    X_raw: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int,
    embed_pca_max: int,
    C_fixed: float,
    random_state: int,
) -> np.ndarray:
    """OOF: her fold’da embed → impute+scale → PCA (fold train) → LR."""
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for tr_pos, ho_pos in skf.split(np.zeros(n), y):
        X_fit = X_raw[tr_pos]
        X_ho = X_raw[ho_pos]
        y_fit = y[tr_pos]
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        Xf = sc.fit_transform(imp.fit_transform(X_fit))
        Xh = sc.transform(imp.transform(X_ho))
        nc = min(embed_pca_max, Xf.shape[1], max(1, Xf.shape[0] - 1))
        pca = PCA(n_components=nc, random_state=random_state)
        Pf = pca.fit_transform(Xf)
        Ph = pca.transform(Xh)
        lr = LogisticRegression(
            C=C_fixed,
            class_weight="balanced",
            max_iter=8000,
            solver="lbfgs",
            random_state=random_state,
        )
        lr.fit(Pf, y_fit)
        oof[ho_pos] = lr.predict_proba(Ph)[:, 1]
    return oof


def _embed_pca_split(
    df: pd.DataFrame,
    tr: pd.Index,
    va: pd.Index,
    te: pd.Index,
    emb_cols: list[str],
    embed_pca_max: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer, StandardScaler, PCA]:
    Xe_tr_raw = df.loc[tr, emb_cols].to_numpy(dtype=np.float64)
    Xe_va_raw = df.loc[va, emb_cols].to_numpy(dtype=np.float64)
    Xe_te_raw = df.loc[te, emb_cols].to_numpy(dtype=np.float64)
    imp_e = SimpleImputer(strategy="median")
    sc_e = StandardScaler()
    Xe_tr = sc_e.fit_transform(imp_e.fit_transform(Xe_tr_raw))
    Xe_va = sc_e.transform(imp_e.transform(Xe_va_raw))
    Xe_te = sc_e.transform(imp_e.transform(Xe_te_raw))
    n_comp = min(embed_pca_max, Xe_tr.shape[1], max(1, Xe_tr.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=random_state)
    return (
        pca.fit_transform(Xe_tr),
        pca.transform(Xe_va),
        pca.transform(Xe_te),
        imp_e,
        sc_e,
        pca,
    )


def tune_hgb_and_threshold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    threshold_objective: str,
    random_state: int,
) -> tuple[dict[str, Any], float, float, HistGradientBoostingClassifier]:
    best: tuple[float, dict[str, Any]] | None = None
    for max_depth in (2, 3, 5):
        for learning_rate in (0.05, 0.1):
            for l2 in (0.5, 2.0, 10.0):
                params = {
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "l2_regularization": l2,
                    "max_iter": 500,
                }
                h = HistGradientBoostingClassifier(
                    **params,
                    random_state=random_state,
                    class_weight="balanced",
                )
                h.fit(X_train, y_train)
                proba = h.predict_proba(X_val)[:, 1]
                _, vscore = pick_threshold(y_val, proba, threshold_objective)
                if best is None or vscore > best[0]:
                    best = (vscore, params)
    assert best is not None
    _, best_params = best
    h_final = HistGradientBoostingClassifier(
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        l2_regularization=best_params["l2_regularization"],
        max_iter=best_params["max_iter"],
        random_state=random_state,
        class_weight="balanced",
    )
    h_final.fit(X_train, y_train)
    proba_val = h_final.predict_proba(X_val)[:, 1]
    thresh, _ = pick_threshold(y_val, proba_val, threshold_objective)
    return best_params, thresh, h_final


def eval_report(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold_default: float,
    threshold_tuned: float,
    *,
    threshold_objective: str,
) -> dict[str, Any]:
    y_hat_def = (proba >= threshold_default).astype(np.int32)
    y_hat_tune = (proba >= threshold_tuned).astype(np.int32)
    sens_t, spec_t = sensitivity_specificity(y_true, y_hat_tune)
    out: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1_positive_threshold_0.5": float(f1_score(y_true, y_hat_def, pos_label=1, zero_division=0)),
        "f1_threshold_tuned": float(f1_score(y_true, y_hat_tune, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat_tune)),
        "sensitivity_tuned": sens_t,
        "specificity_tuned": spec_t,
        "confusion_matrix_tuned": confusion_matrix(y_true, y_hat_tune).tolist(),
        "threshold_tuned": float(threshold_tuned),
        "threshold_objective": threshold_objective,
    }
    return out


def attach_train_val_metrics(
    rep: dict[str, Any],
    *,
    y_train: np.ndarray,
    proba_tr: np.ndarray,
    y_val: np.ndarray,
    proba_val: np.ndarray,
    threshold_tuned: float,
    threshold_objective: str,
) -> None:
    """Aynı eşik (val’da `threshold_objective` ile seçilmiş) train ve val üzerinde değerlendirme."""
    rep["train"] = eval_report(
        y_train, proba_tr, 0.5, threshold_tuned, threshold_objective=threshold_objective
    )
    rep["val"] = eval_report(
        y_val, proba_val, 0.5, threshold_tuned, threshold_objective=threshold_objective
    )


def tune_logreg_c_and_threshold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C_values: list[float],
    *,
    threshold_objective: str,
    random_state: int,
) -> tuple[float, float, LogisticRegression]:
    best: tuple[float, float, float] | None = None  # val_score, C, thresh
    for C in C_values:
        lr = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=8000,
            solver="lbfgs",
            random_state=random_state,
        )
        lr.fit(X_train, y_train)
        proba = lr.predict_proba(X_val)[:, 1]
        t, vscore = pick_threshold(y_val, proba, threshold_objective)
        if best is None or vscore > best[0]:
            best = (vscore, C, t)
    assert best is not None
    _, C_best, _ = best
    lr_final = LogisticRegression(
        C=C_best,
        class_weight="balanced",
        max_iter=8000,
        solver="lbfgs",
        random_state=random_state,
    )
    lr_final.fit(X_train, y_train)
    proba_val = lr_final.predict_proba(X_val)[:, 1]
    thresh, _ = pick_threshold(y_val, proba_val, threshold_objective)
    return C_best, thresh, lr_final


def run_scalar_only(
    df: pd.DataFrame,
    tr: pd.Index,
    va: pd.Index,
    te: pd.Index,
    cols: list[str],
    C_values: list[float],
    *,
    post_block_pca_max: int = 0,
    pca_random_state: int = 0,
    threshold_objective: str,
    random_state: int,
    artifact_mode: str = "scalar_only",
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_train = df.loc[tr, "label"].to_numpy(dtype=np.int32)
    y_val = df.loc[va, "label"].to_numpy(dtype=np.int32)
    y_test = df.loc[te, "label"].to_numpy(dtype=np.int32)
    X_train, X_val, X_test, imp, sc = transform_blocks(
        df.loc[tr, cols].to_numpy(dtype=np.float64),
        df.loc[va, cols].to_numpy(dtype=np.float64),
        df.loc[te, cols].to_numpy(dtype=np.float64),
    )
    pca_block: PCA | None = None
    if post_block_pca_max > 0:
        X_train, X_val, X_test, n_pca, pca_block = pca_fit_transform_train_only(
            X_train, X_val, X_test, post_block_pca_max, random_state=pca_random_state
        )
    else:
        n_pca = 0
    C, thresh, lr = tune_logreg_c_and_threshold(
        X_train,
        y_train,
        X_val,
        y_val,
        C_values,
        threshold_objective=threshold_objective,
        random_state=random_state,
    )
    proba_tr = lr.predict_proba(X_train)[:, 1]
    proba_val = lr.predict_proba(X_val)[:, 1]
    proba_te = lr.predict_proba(X_test)[:, 1]
    rep = eval_report(y_test, proba_te, 0.5, thresh, threshold_objective=threshold_objective)
    attach_train_val_metrics(
        rep,
        y_train=y_train,
        proba_tr=proba_tr,
        y_val=y_val,
        proba_val=proba_val,
        threshold_tuned=thresh,
        threshold_objective=threshold_objective,
    )
    rep["logistic_C"] = C
    rep["n_features"] = len(cols)
    if n_pca > 0:
        rep["post_block_pca_components"] = n_pca
    bundle = {
        "schema_version": 1,
        "mode": artifact_mode,
        "feature_columns": list(cols),
        "imputer": imp,
        "scaler": sc,
        "block_pca": pca_block,
        "classifier": lr,
        "threshold_tuned": float(thresh),
        "threshold_objective": threshold_objective,
    }
    return rep, bundle


def run_with_pca_embed(
    df: pd.DataFrame,
    tr: pd.Index,
    va: pd.Index,
    te: pd.Index,
    sc_cols: list[str],
    emb_cols: list[str],
    embed_pca_max: int,
    C_values: list[float],
    *,
    threshold_objective: str,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_train = df.loc[tr, "label"].to_numpy(dtype=np.int32)
    y_val = df.loc[va, "label"].to_numpy(dtype=np.int32)
    y_test = df.loc[te, "label"].to_numpy(dtype=np.int32)

    Xs_tr, Xs_va, Xs_te, imp_s, sc_s = transform_blocks(
        df.loc[tr, sc_cols].to_numpy(dtype=np.float64),
        df.loc[va, sc_cols].to_numpy(dtype=np.float64),
        df.loc[te, sc_cols].to_numpy(dtype=np.float64),
    )
    Xe_tr_raw = df.loc[tr, emb_cols].to_numpy(dtype=np.float64)
    Xe_va_raw = df.loc[va, emb_cols].to_numpy(dtype=np.float64)
    Xe_te_raw = df.loc[te, emb_cols].to_numpy(dtype=np.float64)
    imp_e = SimpleImputer(strategy="median")
    sc_e = StandardScaler()
    Xe_tr = sc_e.fit_transform(imp_e.fit_transform(Xe_tr_raw))
    Xe_va = sc_e.transform(imp_e.transform(Xe_va_raw))
    Xe_te = sc_e.transform(imp_e.transform(Xe_te_raw))

    n_comp = min(embed_pca_max, Xe_tr.shape[1], max(1, Xe_tr.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=random_state)
    Pe_tr = pca.fit_transform(Xe_tr)
    Pe_va = pca.transform(Xe_va)
    Pe_te = pca.transform(Xe_te)

    X_train = np.hstack([Xs_tr, Pe_tr])
    X_val = np.hstack([Xs_va, Pe_va])
    X_test = np.hstack([Xs_te, Pe_te])

    C, thresh, lr = tune_logreg_c_and_threshold(
        X_train,
        y_train,
        X_val,
        y_val,
        C_values,
        threshold_objective=threshold_objective,
        random_state=random_state,
    )
    proba_tr = lr.predict_proba(X_train)[:, 1]
    proba_val = lr.predict_proba(X_val)[:, 1]
    proba_te = lr.predict_proba(X_test)[:, 1]
    rep = eval_report(y_test, proba_te, 0.5, thresh, threshold_objective=threshold_objective)
    attach_train_val_metrics(
        rep,
        y_train=y_train,
        proba_tr=proba_tr,
        y_val=y_val,
        proba_val=proba_val,
        threshold_tuned=thresh,
        threshold_objective=threshold_objective,
    )
    rep["logistic_C"] = C
    rep["pca_components"] = int(n_comp)
    rep["n_scalar_features"] = len(sc_cols)
    rep["n_embed_doc_features"] = len(emb_cols)
    bundle = {
        "schema_version": 1,
        "mode": "scalar_embed_pca",
        "scalar_columns": list(sc_cols),
        "embed_columns": list(emb_cols),
        "scalar_imputer": imp_s,
        "scalar_scaler": sc_s,
        "embed_imputer": imp_e,
        "embed_scaler": sc_e,
        "embed_pca": pca,
        "classifier": lr,
        "threshold_tuned": float(thresh),
        "threshold_objective": threshold_objective,
    }
    return rep, bundle


def run_early_concat(
    df: pd.DataFrame,
    tr: pd.Index,
    va: pd.Index,
    te: pd.Index,
    sc_cols: list[str],
    ssl_cols: list[str],
    emb_cols: list[str],
    embed_pca_max: int,
    C_values: list[float],
    *,
    ssl_pca_max: int = 0,
    pca_random_state: int = 0,
    threshold_objective: str,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_train = df.loc[tr, "label"].to_numpy(dtype=np.int32)
    y_val = df.loc[va, "label"].to_numpy(dtype=np.int32)
    y_test = df.loc[te, "label"].to_numpy(dtype=np.int32)

    Xs_tr, Xs_va, Xs_te, imp_s, sc_s = transform_blocks(
        df.loc[tr, sc_cols].to_numpy(dtype=np.float64),
        df.loc[va, sc_cols].to_numpy(dtype=np.float64),
        df.loc[te, sc_cols].to_numpy(dtype=np.float64),
    )
    Xl_tr, Xl_va, Xl_te, imp_l, sc_l = transform_blocks(
        df.loc[tr, ssl_cols].to_numpy(dtype=np.float64),
        df.loc[va, ssl_cols].to_numpy(dtype=np.float64),
        df.loc[te, ssl_cols].to_numpy(dtype=np.float64),
    )
    ssl_pca_fitted: PCA | None = None
    if ssl_pca_max > 0:
        Xl_tr, Xl_va, Xl_te, ssl_n, ssl_pca_fitted = pca_fit_transform_train_only(
            Xl_tr, Xl_va, Xl_te, ssl_pca_max, random_state=pca_random_state
        )
    else:
        ssl_n = 0

    blocks_tr = [Xs_tr, Xl_tr]
    blocks_va = [Xs_va, Xl_va]
    blocks_te = [Xs_te, Xl_te]
    embed_n = 0
    imp_e: SimpleImputer | None = None
    sc_e: StandardScaler | None = None
    embed_pca_fitted: PCA | None = None

    if emb_cols:
        Xe_tr_raw = df.loc[tr, emb_cols].to_numpy(dtype=np.float64)
        Xe_va_raw = df.loc[va, emb_cols].to_numpy(dtype=np.float64)
        Xe_te_raw = df.loc[te, emb_cols].to_numpy(dtype=np.float64)
        imp_e = SimpleImputer(strategy="median")
        sc_e = StandardScaler()
        Xe_tr = sc_e.fit_transform(imp_e.fit_transform(Xe_tr_raw))
        Xe_va = sc_e.transform(imp_e.transform(Xe_va_raw))
        Xe_te = sc_e.transform(imp_e.transform(Xe_te_raw))
        embed_n = int(min(embed_pca_max, Xe_tr.shape[1], max(1, Xe_tr.shape[0] - 1)))
        pca = PCA(n_components=embed_n, random_state=pca_random_state)
        blocks_tr.append(pca.fit_transform(Xe_tr))
        blocks_va.append(pca.transform(Xe_va))
        blocks_te.append(pca.transform(Xe_te))
        embed_pca_fitted = pca

    X_train = np.hstack(blocks_tr)
    X_val = np.hstack(blocks_va)
    X_test = np.hstack(blocks_te)

    C, thresh, lr = tune_logreg_c_and_threshold(
        X_train,
        y_train,
        X_val,
        y_val,
        C_values,
        threshold_objective=threshold_objective,
        random_state=random_state,
    )
    proba_tr = lr.predict_proba(X_train)[:, 1]
    proba_val = lr.predict_proba(X_val)[:, 1]
    proba_te = lr.predict_proba(X_test)[:, 1]
    rep = eval_report(y_test, proba_te, 0.5, thresh, threshold_objective=threshold_objective)
    attach_train_val_metrics(
        rep,
        y_train=y_train,
        proba_tr=proba_tr,
        y_val=y_val,
        proba_val=proba_val,
        threshold_tuned=thresh,
        threshold_objective=threshold_objective,
    )
    rep["logistic_C"] = C
    rep["n_scalar"] = len(sc_cols)
    rep["n_ssl"] = len(ssl_cols)
    rep["used_text_embed_pca"] = bool(emb_cols)
    if ssl_n > 0:
        rep["ssl_pca_components"] = ssl_n
    if emb_cols and embed_n > 0:
        rep["embed_pca_components"] = embed_n
    bundle = {
        "schema_version": 1,
        "mode": "early_concat_all",
        "scalar_columns": list(sc_cols),
        "ssl_columns": list(ssl_cols),
        "embed_columns": list(emb_cols),
        "scalar_imputer": imp_s,
        "scalar_scaler": sc_s,
        "ssl_imputer": imp_l,
        "ssl_scaler": sc_l,
        "ssl_pca": ssl_pca_fitted,
        "embed_imputer": imp_e,
        "embed_scaler": sc_e,
        "embed_pca": embed_pca_fitted,
        "classifier": lr,
        "threshold_tuned": float(thresh),
        "threshold_objective": threshold_objective,
    }
    return rep, bundle


def run_hgb_scalar(
    df: pd.DataFrame,
    tr: pd.Index,
    va: pd.Index,
    te: pd.Index,
    sc_cols: list[str],
    *,
    threshold_objective: str,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_train = df.loc[tr, "label"].to_numpy(dtype=np.int32)
    y_val = df.loc[va, "label"].to_numpy(dtype=np.int32)
    y_test = df.loc[te, "label"].to_numpy(dtype=np.int32)
    X_train, X_val, X_test, imp, sc = transform_blocks(
        df.loc[tr, sc_cols].to_numpy(dtype=np.float64),
        df.loc[va, sc_cols].to_numpy(dtype=np.float64),
        df.loc[te, sc_cols].to_numpy(dtype=np.float64),
    )
    hgb_params, thresh, h = tune_hgb_and_threshold(
        X_train, y_train, X_val, y_val, threshold_objective=threshold_objective, random_state=random_state
    )
    proba_tr = h.predict_proba(X_train)[:, 1]
    proba_val = h.predict_proba(X_val)[:, 1]
    proba_te = h.predict_proba(X_test)[:, 1]
    rep = eval_report(y_test, proba_te, 0.5, thresh, threshold_objective=threshold_objective)
    attach_train_val_metrics(
        rep,
        y_train=y_train,
        proba_tr=proba_tr,
        y_val=y_val,
        proba_val=proba_val,
        threshold_tuned=thresh,
        threshold_objective=threshold_objective,
    )
    rep["model"] = "HistGradientBoostingClassifier"
    rep["hgb_params"] = hgb_params
    rep["n_features"] = len(sc_cols)
    bundle = {
        "schema_version": 1,
        "mode": "scalar_hgb",
        "feature_columns": list(sc_cols),
        "imputer": imp,
        "scaler": sc,
        "classifier": h,
        "threshold_tuned": float(thresh),
        "threshold_objective": threshold_objective,
    }
    return rep, bundle


def run_hgb_early_concat(
    df: pd.DataFrame,
    tr: pd.Index,
    va: pd.Index,
    te: pd.Index,
    sc_cols: list[str],
    ssl_cols: list[str],
    emb_cols: list[str],
    embed_pca_max: int,
    *,
    ssl_pca_max: int,
    pca_random_state: int,
    threshold_objective: str,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_train = df.loc[tr, "label"].to_numpy(dtype=np.int32)
    y_val = df.loc[va, "label"].to_numpy(dtype=np.int32)
    y_test = df.loc[te, "label"].to_numpy(dtype=np.int32)

    Xs_tr, Xs_va, Xs_te, imp_s, sc_s = transform_blocks(
        df.loc[tr, sc_cols].to_numpy(dtype=np.float64),
        df.loc[va, sc_cols].to_numpy(dtype=np.float64),
        df.loc[te, sc_cols].to_numpy(dtype=np.float64),
    )
    Xl_tr, Xl_va, Xl_te, imp_l, sc_l = transform_blocks(
        df.loc[tr, ssl_cols].to_numpy(dtype=np.float64),
        df.loc[va, ssl_cols].to_numpy(dtype=np.float64),
        df.loc[te, ssl_cols].to_numpy(dtype=np.float64),
    )
    ssl_pca_fitted: PCA | None = None
    if ssl_pca_max > 0:
        Xl_tr, Xl_va, Xl_te, ssl_n, ssl_pca_fitted = pca_fit_transform_train_only(
            Xl_tr, Xl_va, Xl_te, ssl_pca_max, random_state=pca_random_state
        )
    else:
        ssl_n = 0

    blocks_tr: list[np.ndarray] = [Xs_tr, Xl_tr]
    blocks_va: list[np.ndarray] = [Xs_va, Xl_va]
    blocks_te: list[np.ndarray] = [Xs_te, Xl_te]
    imp_e: SimpleImputer | None = None
    sc_e: StandardScaler | None = None
    emb_p: PCA | None = None
    if emb_cols:
        Pe_tr, Pe_va, Pe_te, imp_e, sc_e, emb_p = _embed_pca_split(
            df, tr, va, te, emb_cols, embed_pca_max, pca_random_state
        )
        blocks_tr.append(Pe_tr)
        blocks_va.append(Pe_va)
        blocks_te.append(Pe_te)

    X_train = np.hstack(blocks_tr)
    X_val = np.hstack(blocks_va)
    X_test = np.hstack(blocks_te)

    hgb_params, thresh, h = tune_hgb_and_threshold(
        X_train, y_train, X_val, y_val, threshold_objective=threshold_objective, random_state=random_state
    )
    proba_tr = h.predict_proba(X_train)[:, 1]
    proba_val = h.predict_proba(X_val)[:, 1]
    proba_te = h.predict_proba(X_test)[:, 1]
    rep = eval_report(y_test, proba_te, 0.5, thresh, threshold_objective=threshold_objective)
    attach_train_val_metrics(
        rep,
        y_train=y_train,
        proba_tr=proba_tr,
        y_val=y_val,
        proba_val=proba_val,
        threshold_tuned=thresh,
        threshold_objective=threshold_objective,
    )
    rep["model"] = "HistGradientBoostingClassifier"
    rep["hgb_params"] = hgb_params
    rep["n_scalar"] = len(sc_cols)
    rep["n_ssl"] = len(ssl_cols)
    rep["used_text_embed_pca"] = bool(emb_cols)
    if ssl_n > 0:
        rep["ssl_pca_components"] = ssl_n
    bundle = {
        "schema_version": 1,
        "mode": "early_concat_hgb",
        "scalar_columns": list(sc_cols),
        "ssl_columns": list(ssl_cols),
        "embed_columns": list(emb_cols),
        "scalar_imputer": imp_s,
        "scalar_scaler": sc_s,
        "ssl_imputer": imp_l,
        "ssl_scaler": sc_l,
        "ssl_pca": ssl_pca_fitted,
        "embed_imputer": imp_e,
        "embed_scaler": sc_e,
        "embed_pca": emb_p,
        "classifier": h,
        "threshold_tuned": float(thresh),
        "threshold_objective": threshold_objective,
    }
    return rep, bundle


def run_late_fusion(
    df: pd.DataFrame,
    tr: pd.Index,
    va: pd.Index,
    te: pd.Index,
    sc_cols: list[str],
    ssl_cols: list[str],
    C_values: list[float],
    *,
    oof_folds: int,
    oof_branch_c: float,
    meta_c: float,
    threshold_objective: str,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_train = df.loc[tr, "label"].to_numpy(dtype=np.int32)
    y_val = df.loc[va, "label"].to_numpy(dtype=np.int32)
    y_test = df.loc[te, "label"].to_numpy(dtype=np.int32)

    Xs_tr, Xs_va, Xs_te, imp_s, sc_s = transform_blocks(
        df.loc[tr, sc_cols].to_numpy(dtype=np.float64),
        df.loc[va, sc_cols].to_numpy(dtype=np.float64),
        df.loc[te, sc_cols].to_numpy(dtype=np.float64),
    )
    Xl_tr, Xl_va, Xl_te, imp_l, sc_l = transform_blocks(
        df.loc[tr, ssl_cols].to_numpy(dtype=np.float64),
        df.loc[va, ssl_cols].to_numpy(dtype=np.float64),
        df.loc[te, ssl_cols].to_numpy(dtype=np.float64),
    )

    C1, _, lr1 = tune_logreg_c_and_threshold(
        Xs_tr, y_train, Xs_va, y_val, C_values, threshold_objective=threshold_objective, random_state=random_state
    )
    C2, _, lr2 = tune_logreg_c_and_threshold(
        Xl_tr, y_train, Xl_va, y_val, C_values, threshold_objective=threshold_objective, random_state=random_state
    )

    Z_va = np.column_stack(
        [lr1.predict_proba(Xs_va)[:, 1], lr2.predict_proba(Xl_va)[:, 1]]
    )
    Z_te = np.column_stack(
        [lr1.predict_proba(Xs_te)[:, 1], lr2.predict_proba(Xl_te)[:, 1]]
    )

    meta = LogisticRegression(
        C=meta_c,
        class_weight="balanced",
        max_iter=8000,
        solver="lbfgs",
        random_state=random_state,
    )
    Z_tr_in = np.column_stack(
        [lr1.predict_proba(Xs_tr)[:, 1], lr2.predict_proba(Xl_tr)[:, 1]]
    )
    meta_train_mode = "in_sample_proba"
    z_scaler = StandardScaler()
    if oof_folds >= 2:
        Xs_tr_raw = df.loc[tr, sc_cols].to_numpy(dtype=np.float64)
        Xl_tr_raw = df.loc[tr, ssl_cols].to_numpy(dtype=np.float64)
        try:
            z1 = oof_branch_positive_proba(
                Xs_tr_raw, y_train, n_folds=oof_folds, C_fixed=oof_branch_c, random_state=random_state
            )
            z2 = oof_branch_positive_proba(
                Xl_tr_raw, y_train, n_folds=oof_folds, C_fixed=oof_branch_c, random_state=random_state + 1
            )
            Z_tr_meta = np.column_stack([z1, z2])
            Z_fit = z_scaler.fit_transform(Z_tr_meta)
            meta.fit(Z_fit, y_train)
            meta_train_mode = "oof_stacking"
        except ValueError:
            Z_fit = z_scaler.fit_transform(Z_tr_in)
            meta.fit(Z_fit, y_train)
            meta_train_mode = "in_sample_proba_oof_failed"
    else:
        Z_fit = z_scaler.fit_transform(Z_tr_in)
        meta.fit(Z_fit, y_train)

    proba_tr = meta.predict_proba(Z_fit)[:, 1]
    Z_va_s = z_scaler.transform(Z_va)
    Z_te_s = z_scaler.transform(Z_te)
    proba_val = meta.predict_proba(Z_va_s)[:, 1]
    thresh, _ = pick_threshold(y_val, proba_val, threshold_objective)
    proba_te = meta.predict_proba(Z_te_s)[:, 1]
    rep = eval_report(y_test, proba_te, 0.5, thresh, threshold_objective=threshold_objective)
    attach_train_val_metrics(
        rep,
        y_train=y_train,
        proba_tr=proba_tr,
        y_val=y_val,
        proba_val=proba_val,
        threshold_tuned=thresh,
        threshold_objective=threshold_objective,
    )
    rep["branch_C_scalar"] = float(C1)
    rep["branch_C_ssl"] = float(C2)
    rep["late_fusion_meta_train"] = meta_train_mode
    rep["late_fusion_oof_folds_requested"] = int(oof_folds)
    rep["late_fusion_oof_used"] = meta_train_mode == "oof_stacking"
    rep["late_fusion_oof_branch_C"] = float(oof_branch_c) if oof_folds >= 2 else None
    rep["late_fusion_meta_C"] = float(meta_c)
    bundle = {
        "schema_version": 1,
        "mode": "late_fusion_logreg",
        "scalar_columns": list(sc_cols),
        "ssl_columns": list(ssl_cols),
        "scalar_imputer": imp_s,
        "scalar_scaler": sc_s,
        "ssl_imputer": imp_l,
        "ssl_scaler": sc_l,
        "branch_scalar_lr": lr1,
        "branch_ssl_lr": lr2,
        "meta": meta,
        "meta_scaler": z_scaler,
        "branch_C_scalar": float(C1),
        "branch_C_ssl": float(C2),
        "late_fusion_meta_C": float(meta_c),
        "late_fusion_oof_folds": int(oof_folds),
        "late_fusion_oof_branch_C": float(oof_branch_c) if oof_folds >= 2 else None,
        "late_fusion_meta_train": meta_train_mode,
        "threshold_tuned": float(thresh),
        "threshold_objective": threshold_objective,
    }
    return rep, bundle


def run_late_fusion3(
    df: pd.DataFrame,
    tr: pd.Index,
    va: pd.Index,
    te: pd.Index,
    sc_cols: list[str],
    ssl_cols: list[str],
    emb_cols: list[str],
    embed_pca_max: int,
    C_values: list[float],
    *,
    oof_folds: int,
    oof_branch_c: float,
    meta_c: float,
    threshold_objective: str,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Skalar | SSL | metin-gömme (PCA+LR) üç dal; meta LR + OOF (mümkünse)."""
    y_train = df.loc[tr, "label"].to_numpy(dtype=np.int32)
    y_val = df.loc[va, "label"].to_numpy(dtype=np.int32)
    y_test = df.loc[te, "label"].to_numpy(dtype=np.int32)

    Xs_tr, Xs_va, Xs_te, imp_s, sc_s = transform_blocks(
        df.loc[tr, sc_cols].to_numpy(dtype=np.float64),
        df.loc[va, sc_cols].to_numpy(dtype=np.float64),
        df.loc[te, sc_cols].to_numpy(dtype=np.float64),
    )
    Xl_tr, Xl_va, Xl_te, imp_l, sc_l = transform_blocks(
        df.loc[tr, ssl_cols].to_numpy(dtype=np.float64),
        df.loc[va, ssl_cols].to_numpy(dtype=np.float64),
        df.loc[te, ssl_cols].to_numpy(dtype=np.float64),
    )
    Pe_tr, Pe_va, Pe_te, imp_e, sc_e, emb_p = _embed_pca_split(
        df, tr, va, te, emb_cols, embed_pca_max, random_state
    )

    C1, _, lr1 = tune_logreg_c_and_threshold(
        Xs_tr, y_train, Xs_va, y_val, C_values, threshold_objective=threshold_objective, random_state=random_state
    )
    C2, _, lr2 = tune_logreg_c_and_threshold(
        Xl_tr, y_train, Xl_va, y_val, C_values, threshold_objective=threshold_objective, random_state=random_state
    )
    C3, _, lr3 = tune_logreg_c_and_threshold(
        Pe_tr, y_train, Pe_va, y_val, C_values, threshold_objective=threshold_objective, random_state=random_state
    )

    Z_va = np.column_stack(
        [
            lr1.predict_proba(Xs_va)[:, 1],
            lr2.predict_proba(Xl_va)[:, 1],
            lr3.predict_proba(Pe_va)[:, 1],
        ]
    )
    Z_te = np.column_stack(
        [
            lr1.predict_proba(Xs_te)[:, 1],
            lr2.predict_proba(Xl_te)[:, 1],
            lr3.predict_proba(Pe_te)[:, 1],
        ]
    )

    meta = LogisticRegression(
        C=meta_c,
        class_weight="balanced",
        max_iter=8000,
        solver="lbfgs",
        random_state=random_state,
    )
    Z_tr_in = np.column_stack(
        [
            lr1.predict_proba(Xs_tr)[:, 1],
            lr2.predict_proba(Xl_tr)[:, 1],
            lr3.predict_proba(Pe_tr)[:, 1],
        ]
    )
    meta_train_mode = "in_sample_proba"
    z_scaler = StandardScaler()
    if oof_folds >= 2:
        Xs_tr_raw = df.loc[tr, sc_cols].to_numpy(dtype=np.float64)
        Xl_tr_raw = df.loc[tr, ssl_cols].to_numpy(dtype=np.float64)
        Xe_tr_raw = df.loc[tr, emb_cols].to_numpy(dtype=np.float64)
        try:
            z1 = oof_branch_positive_proba(
                Xs_tr_raw, y_train, n_folds=oof_folds, C_fixed=oof_branch_c, random_state=random_state
            )
            z2 = oof_branch_positive_proba(
                Xl_tr_raw, y_train, n_folds=oof_folds, C_fixed=oof_branch_c, random_state=random_state + 1
            )
            z3 = oof_text_embed_branch_proba(
                Xe_tr_raw,
                y_train,
                n_folds=oof_folds,
                embed_pca_max=embed_pca_max,
                C_fixed=oof_branch_c,
                random_state=random_state + 2,
            )
            Z_tr_meta = np.column_stack([z1, z2, z3])
            Z_fit = z_scaler.fit_transform(Z_tr_meta)
            meta.fit(Z_fit, y_train)
            meta_train_mode = "oof_stacking"
        except ValueError:
            Z_fit = z_scaler.fit_transform(Z_tr_in)
            meta.fit(Z_fit, y_train)
            meta_train_mode = "in_sample_proba_oof_failed"
    else:
        Z_fit = z_scaler.fit_transform(Z_tr_in)
        meta.fit(Z_fit, y_train)

    proba_tr = meta.predict_proba(Z_fit)[:, 1]
    Z_va_s = z_scaler.transform(Z_va)
    Z_te_s = z_scaler.transform(Z_te)
    proba_val = meta.predict_proba(Z_va_s)[:, 1]
    thresh, _ = pick_threshold(y_val, proba_val, threshold_objective)
    proba_te = meta.predict_proba(Z_te_s)[:, 1]
    rep = eval_report(y_test, proba_te, 0.5, thresh, threshold_objective=threshold_objective)
    attach_train_val_metrics(
        rep,
        y_train=y_train,
        proba_tr=proba_tr,
        y_val=y_val,
        proba_val=proba_val,
        threshold_tuned=thresh,
        threshold_objective=threshold_objective,
    )
    rep["branch_C_scalar"] = float(C1)
    rep["branch_C_ssl"] = float(C2)
    rep["branch_C_text"] = float(C3)
    rep["late_fusion_meta_train"] = meta_train_mode
    rep["late_fusion_branches"] = 3
    rep["late_fusion_oof_folds_requested"] = int(oof_folds)
    rep["late_fusion_oof_used"] = meta_train_mode == "oof_stacking"
    rep["late_fusion_oof_branch_C"] = float(oof_branch_c) if oof_folds >= 2 else None
    rep["late_fusion_meta_C"] = float(meta_c)
    bundle = {
        "schema_version": 1,
        "mode": "late_fusion_logreg3",
        "scalar_columns": list(sc_cols),
        "ssl_columns": list(ssl_cols),
        "embed_columns": list(emb_cols),
        "scalar_imputer": imp_s,
        "scalar_scaler": sc_s,
        "ssl_imputer": imp_l,
        "ssl_scaler": sc_l,
        "embed_imputer": imp_e,
        "embed_scaler": sc_e,
        "embed_pca": emb_p,
        "branch_scalar_lr": lr1,
        "branch_ssl_lr": lr2,
        "branch_text_lr": lr3,
        "meta": meta,
        "meta_scaler": z_scaler,
        "branch_C_scalar": float(C1),
        "branch_C_ssl": float(C2),
        "branch_C_text": float(C3),
        "late_fusion_meta_C": float(meta_c),
        "late_fusion_oof_folds": int(oof_folds),
        "late_fusion_oof_branch_C": float(oof_branch_c) if oof_folds >= 2 else None,
        "late_fusion_meta_train": meta_train_mode,
        "threshold_tuned": float(thresh),
        "threshold_objective": threshold_objective,
    }
    return rep, bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="benchmark_v2 train / değerlendirme")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--modes",
        default="scalar_hgb",
        help="virgülle ayrılmış mod listesi (varsayılan: yalnızca API ile uyumlu scalar_hgb)",
    )
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="eğitilen sklearn nesnelerini experiments/benchmark_v2/outputs/models/<mod>.joblib olarak kaydet",
    )
    parser.add_argument(
        "--copy-to-api-model",
        action="store_true",
        help="scalar_hgb.joblib oluştuysa inference_api/models/model.joblib üzerine kopyala",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config) if args.config.is_file() else {}
    ev = cfg.get("eval") or {}
    seed = int(cfg.get("random_seed", 42))
    pca_max_legacy = int(ev.get("pca_max_components", 16))
    embed_pca_max = int(ev.get("embed_pca_max_components", 48))
    ssl_pca_max = int(ev.get("ssl_pca_components", 128))
    oof_folds = int(ev.get("late_fusion_oof_folds", 5))
    oof_branch_c = float(ev.get("late_fusion_oof_branch_c", 1.0))
    meta_c = float(ev.get("late_fusion_meta_C", 1.0))
    threshold_objective = str(ev.get("threshold_objective", "balanced_accuracy"))
    C_values = [float(x) for x in ev.get("logistic_c_values", [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0])]

    if not args.merged.is_file():
        print(f"merged özellik yok: {args.merged}", file=sys.stderr)
        return 1
    if not args.splits.is_file():
        print(f"split dosyası yok: {args.splits}", file=sys.stderr)
        return 1

    ensure_benchmark_output_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    merged = (
        pd.read_parquet(args.merged) if args.merged.suffix.lower() == ".parquet" else pd.read_csv(args.merged)
    )
    splits = (
        pd.read_parquet(args.splits) if args.splits.suffix.lower() == ".parquet" else pd.read_csv(args.splits)
    )

    df = pd.merge(merged, splits[["source_relpath", "split"]], on="source_relpath", how="inner")
    if "label" not in df.columns:
        df = pd.merge(df, splits[["source_relpath", "label"]], on="source_relpath", how="left")

    tr = df.index[df["split"] == "train"]
    va = df.index[df["split"] == "val"]
    te = df.index[df["split"] == "test"]
    if len(tr) < 5 or len(va) < 2 or len(te) < 2:
        print(f"Yetersiz örnek: train={len(tr)} val={len(va)} test={len(te)}", file=sys.stderr)
        return 1

    sc_candidates = scalar_columns(df)
    ssl_candidates = ssl_columns(df)
    emb_candidates = embed_doc_columns(df)
    sc_cols = columns_observed_in_train(df, tr, sc_candidates)
    ssl_cols = columns_observed_in_train(df, tr, ssl_candidates)
    emb_cols = columns_observed_in_train(df, tr, emb_candidates)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "merged": str(args.merged),
        "splits": str(args.splits),
        "n_total": int(len(df)),
        "feature_columns": {
            "scalar": {"candidates": len(sc_candidates), "used_after_train_non_nan_filter": len(sc_cols)},
            "ssl": {"candidates": len(ssl_candidates), "used_after_train_non_nan_filter": len(ssl_cols)},
            "embed_doc": {"candidates": len(emb_candidates), "used_after_train_non_nan_filter": len(emb_cols)},
        },
        "eval_settings": {
            "random_seed": seed,
            "threshold_objective": threshold_objective,
            "embed_pca_max_components": embed_pca_max,
            "pca_max_components_legacy": pca_max_legacy,
            "ssl_pca_components": ssl_pca_max,
            "late_fusion_oof_folds": oof_folds,
            "late_fusion_oof_branch_C": oof_branch_c if oof_folds >= 2 else None,
            "late_fusion_meta_C": meta_c,
            "save_models": bool(args.save_models),
        },
        "models_dir": str(MODELS_DIR.resolve()) if args.save_models else None,
        "modes": {},
    }

    for mode in modes:
        try:
            if mode == "scalar_only":
                if len(sc_cols) < 2:
                    results["modes"][mode] = {"error": "scalar sütun yetersiz"}
                    continue
                rep, bundle = run_scalar_only(
                    df,
                    tr,
                    va,
                    te,
                    sc_cols,
                    C_values,
                    post_block_pca_max=0,
                    pca_random_state=seed,
                    threshold_objective=threshold_objective,
                    random_state=seed,
                )
            elif mode == "scalar_hgb":
                if len(sc_cols) < 2:
                    results["modes"][mode] = {"error": "scalar sütun yetersiz"}
                    continue
                rep, bundle = run_hgb_scalar(
                    df, tr, va, te, sc_cols, threshold_objective=threshold_objective, random_state=seed
                )
            elif mode == "scalar_embed_pca":
                if len(sc_cols) < 2:
                    results["modes"][mode] = {"error": "scalar sütun yetersiz"}
                    continue
                if not emb_cols:
                    results["modes"][mode] = {"error": "embed_doc_* yok; 03 ile --with-text-embeddings çalıştırın"}
                    continue
                rep, bundle = run_with_pca_embed(
                    df,
                    tr,
                    va,
                    te,
                    sc_cols,
                    emb_cols,
                    embed_pca_max,
                    C_values,
                    threshold_objective=threshold_objective,
                    random_state=seed,
                )
            elif mode == "ssl_only":
                if len(ssl_cols) < 2:
                    results["modes"][mode] = {"error": "ssl_dim_* yetersiz"}
                    continue
                rep, bundle = run_scalar_only(
                    df,
                    tr,
                    va,
                    te,
                    ssl_cols,
                    C_values,
                    post_block_pca_max=ssl_pca_max,
                    pca_random_state=seed,
                    threshold_objective=threshold_objective,
                    random_state=seed,
                    artifact_mode="ssl_only",
                )
            elif mode == "early_concat_all":
                if len(sc_cols) < 2 or len(ssl_cols) < 2:
                    results["modes"][mode] = {"error": "scalar veya ssl sütun yetersiz"}
                    continue
                rep, bundle = run_early_concat(
                    df,
                    tr,
                    va,
                    te,
                    sc_cols,
                    ssl_cols,
                    emb_cols,
                    embed_pca_max,
                    C_values,
                    ssl_pca_max=ssl_pca_max,
                    pca_random_state=seed,
                    threshold_objective=threshold_objective,
                    random_state=seed,
                )
            elif mode == "early_concat_hgb":
                if len(sc_cols) < 2 or len(ssl_cols) < 2:
                    results["modes"][mode] = {"error": "scalar veya ssl sütun yetersiz"}
                    continue
                rep, bundle = run_hgb_early_concat(
                    df,
                    tr,
                    va,
                    te,
                    sc_cols,
                    ssl_cols,
                    emb_cols,
                    embed_pca_max,
                    ssl_pca_max=ssl_pca_max,
                    pca_random_state=seed,
                    threshold_objective=threshold_objective,
                    random_state=seed,
                )
            elif mode == "late_fusion_logreg":
                if len(sc_cols) < 2 or len(ssl_cols) < 2:
                    results["modes"][mode] = {"error": "scalar veya ssl sütun yetersiz"}
                    continue
                rep, bundle = run_late_fusion(
                    df,
                    tr,
                    va,
                    te,
                    sc_cols,
                    ssl_cols,
                    C_values,
                    oof_folds=oof_folds,
                    oof_branch_c=oof_branch_c,
                    meta_c=meta_c,
                    threshold_objective=threshold_objective,
                    random_state=seed,
                )
            elif mode == "late_fusion_logreg3":
                if len(sc_cols) < 2 or len(ssl_cols) < 2:
                    results["modes"][mode] = {"error": "scalar veya ssl sütun yetersiz"}
                    continue
                if not emb_cols:
                    results["modes"][mode] = {"error": "embed_doc_* yok; 03 --with-text-embeddings gerekli"}
                    continue
                rep, bundle = run_late_fusion3(
                    df,
                    tr,
                    va,
                    te,
                    sc_cols,
                    ssl_cols,
                    emb_cols,
                    embed_pca_max,
                    C_values,
                    oof_folds=oof_folds,
                    oof_branch_c=oof_branch_c,
                    meta_c=meta_c,
                    threshold_objective=threshold_objective,
                    random_state=seed,
                )
            else:
                results["modes"][mode] = {"error": f"bilinmeyen mod: {mode}"}
                continue

            if args.save_models:
                model_path = MODELS_DIR / f"{mode}.joblib"
                joblib.dump(bundle, model_path)
                rep = dict(rep)
                rep["saved_model_path"] = str(model_path.resolve())
            results["modes"][mode] = rep
        except Exception as e:
            results["modes"][mode] = {"error": str(e)}

    out_path = args.output or (
        REPORTS_DIR / f"benchmark_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rapor: {out_path}")

    if args.copy_to_api_model and args.save_models:
        src = MODELS_DIR / "scalar_hgb.joblib"
        if src.is_file():
            API_MODEL_JOBLIB.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, API_MODEL_JOBLIB)
            print(f"API modeli güncellendi: {API_MODEL_JOBLIB}")
        else:
            print(
                f"[uyarı] --copy-to-api-model: {src} yok (scalar_hgb modu çalışmadı veya hata).",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

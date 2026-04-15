# =========================================================
# TEST-RETEST INFERENCE + PAIR CONSISTENCY ANALYSIS
# =========================================================

from google.colab import drive
drive.mount('/content/drive')

!pip -q install transformers torchaudio librosa soundfile audioread
!apt-get -qq install -y ffmpeg

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import librosa
from transformers import AutoFeatureExtractor, AutoModel
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

MODEL_NAME = "facebook/wav2vec2-base"
MODEL_DIR = "/content/drive/MyDrive/Huseyin/depression/depression_voice_model_results_v4_segmented"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "best_threshold.json")

TEST_RETEST_DIR = Path("/content/drive/MyDrive/Diagvoice YK/Voice Records/Test-retest (Önemli Veri!!!)/öğrenci reliability")

TARGET_SR = 16000
SEGMENT_SEC = 8
SEGMENT_LEN = TARGET_SR * SEGMENT_SEC
HOP_SEC = 4
HOP_LEN = TARGET_SR * HOP_SEC
TOP_DB = 25
MIN_KEEP_SEC = 2.0
MAX_SEGMENTS_PER_FILE_EVAL = 20
BATCH_SIZE = 4

ALLOWED_EXTS = {".wav", ".m4a", ".mp3", ".aac", ".flac", ".ogg", ".mp4"}

OUT_SEGMENT_CSV = os.path.join(MODEL_DIR, "test_retest_segment_predictions.csv")
OUT_FILE_CSV = os.path.join(MODEL_DIR, "test_retest_file_predictions.csv")
OUT_PAIR_CSV = os.path.join(MODEL_DIR, "test_retest_pair_analysis.csv")

# =========================================================
# LOAD THRESHOLD
# =========================================================
if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH, "r") as f:
        best_threshold = float(json.load(f)["threshold"])
else:
    best_threshold = 0.5

print("Using threshold:", best_threshold)

# =========================================================
# HELPERS
# =========================================================
def safe_stem(path):
    return Path(path).stem.strip()

def normalize_text(s):
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9ğüşöçıİĞÜŞÖÇ_]+", "", s)
    return s.strip("_")

def extract_student_id(path_str):
    """
    .../OGRENCI10/Fatma 1.m4a -> OGRENCI10
    """
    parts = Path(path_str).parts
    for p in parts:
        if re.match(r"^OGRENCI\d+$", p.upper()):
            return p.upper()
    return "UNKNOWN_STUDENT"

def extract_recording_order(filename):
    """
    Fatma 1.m4a -> 1
    Fatma 2.m4a -> 2
    AUDIO-2025-10-22-13-20-46.m4a -> None
    """
    stem = safe_stem(filename)
    m = re.search(r"(?:^|\s|_)(\d+)$", stem)
    if m:
        return int(m.group(1))
    return None

def parse_datetime_from_filename(filename):
    """
    AUDIO-2025-10-22-13-20-46.m4a -> datetime
    """
    stem = safe_stem(filename)

    patterns = [
        r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})",
        r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})",
        r"(\d{8}_\d{6})",
    ]

    for pat in patterns:
        m = re.search(pat, stem)
        if m:
            txt = m.group(1)
            for fmt in ["%Y-%m-%d-%H-%M-%S", "%Y_%m_%d_%H_%M_%S", "%Y%m%d_%H%M%S"]:
                try:
                    return datetime.strptime(txt, fmt)
                except:
                    pass
    return None

def build_recording_key(path_str):
    p = Path(path_str)
    student_id = extract_student_id(path_str)
    return f"{student_id}__{normalize_text(p.stem)}"

def load_and_preprocess_audio(path, target_sr=16000, top_db=25, min_keep_sec=2.0):
    """
    m4a/wav/mp3 gibi formatlar için librosa.load kullanıyoruz.
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    y = y.astype(np.float32)
    y = np.nan_to_num(y)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    y = y - np.mean(y)

    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)

    if len(y_trimmed) < int(min_keep_sec * sr):
        y_trimmed = y

    peak = np.max(np.abs(y_trimmed)) + 1e-8
    y_trimmed = y_trimmed / peak

    return y_trimmed.astype(np.float32), sr, len(y) / sr, len(y_trimmed) / sr

def make_segments(y, sr=16000, segment_len=16000*8, hop_len=16000*4, max_segments=None):
    if len(y) <= segment_len:
        pad = segment_len - len(y)
        seg = np.pad(y, (0, pad), mode="constant")
        return [seg.astype(np.float32)]

    starts = list(range(0, len(y) - segment_len + 1, hop_len))
    last_start = len(y) - segment_len

    if len(starts) == 0 or starts[-1] != last_start:
        starts.append(last_start)

    if max_segments is not None and len(starts) > max_segments:
        idxs = np.linspace(0, len(starts) - 1, max_segments, dtype=int)
        starts = [starts[i] for i in idxs]

    segments = []
    for s in starts:
        seg = y[s:s + segment_len]
        segments.append(seg.astype(np.float32))

    return segments

# =========================================================
# COLLECT AUDIO FILES
# =========================================================
audio_files = sorted([p for p in TEST_RETEST_DIR.rglob("*") if p.suffix.lower() in ALLOWED_EXTS])

rows = []
for f in audio_files:
    rows.append({
        "path": str(f),
        "filename": f.name,
        "student_id": extract_student_id(str(f)),
        "recording_key": build_recording_key(str(f)),
        "recording_order": extract_recording_order(f.name),
        "recording_datetime": parse_datetime_from_filename(f.name)
    })

file_df = pd.DataFrame(rows)

print("Found audio files:", len(file_df))
print(file_df.head())

# =========================================================
# BUILD SEGMENT TABLE
# =========================================================
seg_rows = []

for _, row in tqdm(file_df.iterrows(), total=len(file_df), desc="Preparing segments"):
    try:
        y, sr, dur_before, dur_after = load_and_preprocess_audio(
            row["path"],
            target_sr=TARGET_SR,
            top_db=TOP_DB,
            min_keep_sec=MIN_KEEP_SEC
        )

        segs = make_segments(
            y,
            sr=sr,
            segment_len=SEGMENT_LEN,
            hop_len=HOP_LEN,
            max_segments=MAX_SEGMENTS_PER_FILE_EVAL
        )

        for seg_idx, seg in enumerate(segs):
            seg_rows.append({
                "path": row["path"],
                "filename": row["filename"],
                "student_id": row["student_id"],
                "recording_key": row["recording_key"],
                "recording_order": row["recording_order"],
                "recording_datetime": row["recording_datetime"],
                "segment_idx": seg_idx,
                "num_segments_in_file": len(segs),
                "duration_before_trim_sec": dur_before,
                "duration_after_trim_sec": dur_after,
                "audio": seg
            })

    except Exception as e:
        print(f"ERROR processing {row['path']}: {e}")

seg_df = pd.DataFrame(seg_rows)
print("Total segments:", len(seg_df))

# =========================================================
# FEATURE EXTRACTOR
# =========================================================
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

# =========================================================
# DATASET / COLLATOR
# =========================================================
class InferenceSegmentDataset(Dataset):
    def __init__(self, seg_df):
        self.df = seg_df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "input_values": row["audio"],
            "path": row["path"],
            "filename": row["filename"],
            "student_id": row["student_id"],
            "recording_key": row["recording_key"],
            "recording_order": row["recording_order"],
            "recording_datetime": row["recording_datetime"],
            "segment_idx": int(row["segment_idx"]),
            "num_segments_in_file": int(row["num_segments_in_file"]),
            "duration_before_trim_sec": float(row["duration_before_trim_sec"]),
            "duration_after_trim_sec": float(row["duration_after_trim_sec"]),
        }

@dataclass
class Collator:
    feature_extractor: any

    def __call__(self, batch):
        audios = [item["input_values"] for item in batch]

        inputs = self.feature_extractor(
            audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True
        )

        return {
            "input_values": inputs["input_values"],
            "attention_mask": inputs.get("attention_mask"),
            "paths": [item["path"] for item in batch],
            "filenames": [item["filename"] for item in batch],
            "student_ids": [item["student_id"] for item in batch],
            "recording_keys": [item["recording_key"] for item in batch],
            "recording_orders": [item["recording_order"] for item in batch],
            "recording_datetimes": [item["recording_datetime"] for item in batch],
            "segment_idxs": [item["segment_idx"] for item in batch],
            "num_segments_in_file": [item["num_segments_in_file"] for item in batch],
            "duration_before_trim_sec": [item["duration_before_trim_sec"] for item in batch],
            "duration_after_trim_sec": [item["duration_after_trim_sec"] for item in batch],
        }

dataset = InferenceSegmentDataset(seg_df)
collator = Collator(feature_extractor)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
    num_workers=0,
    pin_memory=True
)

# =========================================================
# MODEL
# =========================================================
class Wav2VecBinaryClassifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def masked_mean_pooling(self, hidden_states, attention_mask):
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.float()
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(1),
            size=hidden_states.size(1),
            mode="nearest"
        ).squeeze(1)

        mask = mask.unsqueeze(-1)
        summed = torch.sum(hidden_states * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / denom

    def forward(self, input_values, attention_mask=None):
        outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = self.masked_mean_pooling(hidden_states, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        return logits

model = Wav2VecBinaryClassifier(MODEL_NAME, dropout=0.3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded from:", MODEL_PATH)

# =========================================================
# INFERENCE
# =========================================================
segment_pred_rows = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Running inference"):
        input_values = batch["input_values"].to(DEVICE)
        attention_mask = batch["attention_mask"]

        if attention_mask is not None:
            attention_mask = attention_mask.to(DEVICE)

        logits = model(input_values=input_values, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= best_threshold).long()

        for i in range(len(batch["paths"])):
            pred_label = int(preds[i].cpu().item())
            segment_pred_rows.append({
                "path": batch["paths"][i],
                "filename": batch["filenames"][i],
                "student_id": batch["student_ids"][i],
                "recording_key": batch["recording_keys"][i],
                "recording_order": batch["recording_orders"][i],
                "recording_datetime": batch["recording_datetimes"][i],
                "segment_idx": batch["segment_idxs"][i],
                "num_segments_in_file": batch["num_segments_in_file"][i],
                "duration_before_trim_sec": batch["duration_before_trim_sec"][i],
                "duration_after_trim_sec": batch["duration_after_trim_sec"][i],
                "prob_depression": float(probs[i].cpu().item()),
                "segment_pred_label": pred_label,
                "segment_pred_name": "depresyon" if pred_label == 1 else "saglikli"
            })

segment_pred_df = pd.DataFrame(segment_pred_rows)
segment_pred_df.to_csv(OUT_SEGMENT_CSV, index=False)

# =========================================================
# FILE-LEVEL AGGREGATION
# =========================================================
file_pred_df = segment_pred_df.groupby(
    ["path", "filename", "student_id", "recording_key", "recording_order", "recording_datetime"],
    as_index=False,
    dropna=False
).agg({
    "prob_depression": ["mean", "max", "count"],
    "duration_before_trim_sec": "first",
    "duration_after_trim_sec": "first"
})

file_pred_df.columns = [
    "path", "filename", "student_id", "recording_key", "recording_order", "recording_datetime",
    "mean_prob_depression", "max_prob_depression", "segment_count",
    "duration_before_trim_sec", "duration_after_trim_sec"
]

file_pred_df["pred_label"] = (file_pred_df["mean_prob_depression"] >= best_threshold).astype(int)
file_pred_df["pred_label_name"] = file_pred_df["pred_label"].map({0: "saglikli", 1: "depresyon"})
file_pred_df["used_threshold"] = best_threshold
file_pred_df["recording_datetime_str"] = file_pred_df["recording_datetime"].astype(str)

file_pred_df = file_pred_df.sort_values(
    ["student_id", "recording_order", "recording_datetime_str", "filename"],
    na_position="last"
).reset_index(drop=True)

file_pred_df.to_csv(OUT_FILE_CSV, index=False)

print("File-level row count:", len(file_pred_df))
print(file_pred_df[["student_id", "filename", "recording_order", "recording_datetime"]].head(20))

# =========================================================
# PAIR ANALYSIS
# =========================================================
def compute_time_diff_minutes(dt1, dt2):
    if pd.isna(dt1) or pd.isna(dt2) or dt1 is None or dt2 is None:
        return np.nan
    return abs((dt2 - dt1).total_seconds()) / 60.0

pair_rows = []

for student_id, group in file_pred_df.groupby("student_id", dropna=False):
    group = group.copy()

    has_order = group["recording_order"].notna().sum() >= 2
    has_datetime = group["recording_datetime"].notna().sum() >= 2

    if has_order:
        group = group.sort_values(["recording_order", "filename"], na_position="last")
    elif has_datetime:
        group = group.sort_values(["recording_datetime", "filename"], na_position="last")
    else:
        group = group.sort_values(["filename"], na_position="last")

    group = group.reset_index(drop=True)

    if len(group) < 2:
        continue

    # her öğrenci için ilk iki kaydı eşleştir
    r1 = group.iloc[0]
    r2 = group.iloc[1]

    time_diff_min = compute_time_diff_minutes(r1["recording_datetime"], r2["recording_datetime"])
    same_label = int(r1["pred_label"] == r2["pred_label"])
    prob_diff = abs(float(r1["mean_prob_depression"]) - float(r2["mean_prob_depression"]))
    mean_pair_prob = (float(r1["mean_prob_depression"]) + float(r2["mean_prob_depression"])) / 2.0

    if same_label == 1 and prob_diff < 0.10:
        consistency = "yüksek_tutarlılık"
    elif same_label == 1 and prob_diff < 0.20:
        consistency = "orta_tutarlılık"
    else:
        consistency = "düşük_tutarlılık"

    pair_rows.append({
        "student_id": student_id,

        "file1_path": r1["path"],
        "file1_name": r1["filename"],
        "file1_datetime": r1["recording_datetime"],
        "file1_order": r1["recording_order"],
        "file1_mean_prob_depression": float(r1["mean_prob_depression"]),
        "file1_pred_label": int(r1["pred_label"]),
        "file1_pred_name": r1["pred_label_name"],

        "file2_path": r2["path"],
        "file2_name": r2["filename"],
        "file2_datetime": r2["recording_datetime"],
        "file2_order": r2["recording_order"],
        "file2_mean_prob_depression": float(r2["mean_prob_depression"]),
        "file2_pred_label": int(r2["pred_label"]),
        "file2_pred_name": r2["pred_label_name"],

        "time_diff_minutes": time_diff_min,
        "same_pred_label": same_label,
        "probability_absolute_diff": prob_diff,
        "pair_mean_probability": mean_pair_prob,
        "consistency_comment": consistency
    })

pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(OUT_PAIR_CSV, index=False)

print("Pair row count:", len(pair_df))
print(pair_df.head(20))

# =========================================================
# SUMMARY
# =========================================================
print("\n================ FILE-LEVEL SUMMARY ================")
print(file_pred_df["pred_label_name"].value_counts(dropna=False))
print(file_pred_df.head(20))

print("\n================ PAIR SUMMARY ================")
if len(pair_df) > 0:
    print(pair_df["consistency_comment"].value_counts(dropna=False))
    print(pair_df.head(20))
else:
    print("No pair rows created. Öğrenci klasörlerinde eşleşecek yeterli dosya bulunamadı.")

print("\nSaved segment-level CSV :", OUT_SEGMENT_CSV)
print("Saved file-level CSV    :", OUT_FILE_CSV)
print("Saved pair-level CSV    :", OUT_PAIR_CSV)
import io
import os
import tempfile
import traceback

import fastapi
import librosa
import torch
import torch.nn as nn
import uvicorn
from fastapi import File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from transformers import AutoModel


EMOTIONS = ["sadness", "fear", "happiness", "anger"]
ID2LABEL = {0: "sadness", 1: "fear", 2: "happiness", 3: "anger"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

TARGET_SR = 16000
MAX_SEC = 10.0
MAX_SAMPLES = int(TARGET_SR * MAX_SEC)

N_SEGMENTS = int(os.environ.get("AUDIO_N_SEGMENTS", "5"))
N_SEGMENTS = max(1, min(20, N_SEGMENTS))

ENABLE_VAD = os.environ.get("AUDIO_ENABLE_VAD", "1").strip() not in {"0", "false", "False"}
VAD_TOP_DB = float(os.environ.get("AUDIO_VAD_TOP_DB", "25"))
VAD_MAX_KEEP_SEC = float(os.environ.get("AUDIO_VAD_MAX_KEEP_SEC", "30"))
VAD_MAX_KEEP_SEC = max(1.0, min(120.0, VAD_MAX_KEEP_SEC))


class AttentivePool(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        scores = self.net(hs).squeeze(-1)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (hs * w).sum(dim=1)


class WavLMMultiTask(nn.Module):
    def __init__(self, model_name: str, n_classes: int = 4, dropout: float = 0.35):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.pool = AttentivePool(hidden)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(hidden, n_classes)
        self.int_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_values: torch.Tensor):
        out = self.backbone(input_values=input_values)
        hs = out.last_hidden_state
        pooled = self.pool(hs)
        pooled = self.dropout(pooled)
        logits = self.cls_head(pooled)
        inten = self.int_head(pooled).squeeze(-1)
        return logits, inten


def load_checkpoint(model: nn.Module, checkpoint_path: str):
    obj = torch.load(checkpoint_path, map_location="cpu")
    state_dict = None
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict):
        state_dict = obj

    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain a state_dict-like object")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected


app = fastapi.FastAPI()

_cors_origins_raw = os.environ.get(
    "CORS_ALLOW_ORIGINS",
    "https://diagvoice.huseyinalav.me,http://localhost:5173,http://127.0.0.1:5173","https://duygudurum.diagvoice.com",
)
_cors_allow_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("AUDIO_SENTIMENT_CKPT", "wavlm_v6_multicrop_ema_best.pt")
BACKBONE_NAME = os.environ.get("AUDIO_SENTIMENT_BACKBONE", "microsoft/wavlm-base-plus")

print("Starting voice-to-sentiment server and loading model...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = WavLMMultiTask(model_name=BACKBONE_NAME, n_classes=4, dropout=0.35)
    missing_keys, unexpected_keys = load_checkpoint(model, MODEL_PATH)
    print(f"Checkpoint loaded: {MODEL_PATH}")
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    model.to(device)
    model.eval()
except Exception:
    print("!!! FATAL: Failed to load audio sentiment model !!!")
    print(traceback.format_exc())
    model = None
    device = "cpu"


def decode_audio_to_16k_mono_float32(audio_bytes: bytes, filename: str):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        audio_segment = AudioSegment.from_file(tmp_path)
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        y, sr = librosa.load(wav_buffer, sr=TARGET_SR, mono=True)
        if y is None or len(y) == 0:
            raise RuntimeError("Empty audio")
        return y.astype("float32"), sr
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def decode_audio_file_to_16k_mono_float32(path: str):
    audio_segment = AudioSegment.from_file(path)
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    y, sr = librosa.load(wav_buffer, sr=TARGET_SR, mono=True)
    if y is None or len(y) == 0:
        raise RuntimeError("Empty audio")
    return y.astype("float32"), sr


def preprocess_waveform(y, sr: int):
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    if len(y) >= MAX_SAMPLES:
        start = (len(y) - MAX_SAMPLES) // 2
        y = y[start : start + MAX_SAMPLES]
    else:
        pad = MAX_SAMPLES - len(y)
        y = torch.nn.functional.pad(torch.from_numpy(y), (0, pad)).numpy()

    m = float(abs(y).max() + 1e-6)
    y = (y / m).astype("float32")
    return y, sr


def apply_vad(y, sr: int):
    if not ENABLE_VAD:
        return y
    try:
        intervals = librosa.effects.split(y, top_db=VAD_TOP_DB)
        if intervals is None or len(intervals) == 0:
            return y

        chunks = []
        total = 0
        max_keep = int(sr * VAD_MAX_KEEP_SEC)
        for start, end in intervals:
            if end <= start:
                continue
            seg = y[start:end]
            if len(seg) == 0:
                continue
            remaining = max_keep - total
            if remaining <= 0:
                break
            if len(seg) > remaining:
                seg = seg[:remaining]
            chunks.append(seg)
            total += len(seg)

        if total <= 0:
            return y

        y2 = torch.cat([torch.from_numpy(c) for c in chunks]).numpy()
        if len(y2) < int(0.2 * sr):
            return y
        return y2.astype("float32")
    except Exception:
        return y


def make_crops(y, sr: int):
    if len(y) <= MAX_SAMPLES:
        y1, _ = preprocess_waveform(y, sr)
        return [y1]

    n = min(N_SEGMENTS, max(1, len(y) // MAX_SAMPLES))
    if n <= 1:
        y1, _ = preprocess_waveform(y, sr)
        return [y1]

    max_start = len(y) - MAX_SAMPLES
    if max_start <= 0:
        y1, _ = preprocess_waveform(y, sr)
        return [y1]

    starts = [int(round(i * (max_start / (n - 1)))) for i in range(n)]
    crops = []
    for st in starts:
        seg = y[st : st + MAX_SAMPLES]
        y1, _ = preprocess_waveform(seg, sr)
        crops.append(y1)
    return crops


@app.get("/")
async def root():
    return {
        "message": "Voice-to-Sentiment API. Send audio to /analyze-audio (multipart/form-data file).",
        "num_classes": 4,
        "labels": ID2LABEL,
    }


@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model could not be loaded. Check server logs.")

    try:
        tmp_path = None
        suffix = ""
        if file.filename:
            safe = file.filename.replace("/", "_").replace("\\", "_")
            suffix = f"_{safe}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)

        y, sr = decode_audio_file_to_16k_mono_float32(tmp_path)
        y = apply_vad(y, sr)

        crops = make_crops(y, sr)
        x = torch.stack([torch.from_numpy(c) for c in crops], dim=0).to(device)
        with torch.no_grad():
            logits, intensity = model(x)
            probs_batch = torch.softmax(logits, dim=-1).detach().cpu()
            intensity_batch = intensity.detach().cpu()

            probs_mean = probs_batch.mean(dim=0)
            probs = probs_mean.tolist()
            pred_id = int(torch.argmax(probs_mean, dim=-1).item())
            intensity_value = float(intensity_batch.mean().item())

        return {
            "pred_id": pred_id,
            "pred_label": ID2LABEL.get(pred_id),
            "probs": probs,
            "intensity": intensity_value,
            "sample_rate": sr,
            "duration_sec": float(len(y) / sr),
            "num_segments": int(len(crops)),
            "labels": ID2LABEL,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {e}")
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

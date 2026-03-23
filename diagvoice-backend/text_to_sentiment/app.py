from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import os
from fastapi.middleware.cors import CORSMiddleware

import io
import numpy as np
import torch
import torchaudio
import joblib

# =========================
# FastAPI
# =========================
app = FastAPI()

_cors_origins_raw = os.environ.get(
    "CORS_ALLOW_ORIGINS",
    "https://diagvoice.huseyinalav.me,http://localhost:5173,http://127.0.0.1:5173,https://duygudurum.diagvoice.com,https://algoritmalar.diagvoice.com",
)
_cors_allow_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Text Sentiment Model
# =========================
sentiment_pipeline = pipeline("sentiment-analysis", model="kaixkhazaki/turkish-sentiment")

class SentimentRequest(BaseModel):
    text: str

def text_to_valence(label: str, score: float) -> float:
    """
    label: POSITIVE/NEGATIVE (veya Positive/Negative)
    score: seçilen label güveni (0..1)
    """
    lab = str(label).upper()
    s = float(score)
    if lab == "POSITIVE":
        return s
    # NEGATIVE ise
    return 1.0 - s

# =========================
# Audio Valence Model (wav2vec2 + joblib regressor)
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_MODEL_PATH = os.environ.get(
    "AUDIO_VALENCE_MODEL_PATH",
    "/app/models/audio_valence_model.joblib"
)

# wav2vec2 torchaudio bundle
WAV2VEC_BUNDLE = torchaudio.pipelines.WAV2VEC2_BASE
wav2vec_model = None
audio_regressor = None

@app.on_event("startup")
def _load_audio_models():
    global wav2vec_model, audio_regressor
    # wav2vec2
    wav2vec_model = WAV2VEC_BUNDLE.get_model().to(DEVICE).eval()
    # sklearn pipeline (StandardScaler + Ridge) - opsiyonel; yoksa /analyze_fusion çalışmaz
    if os.path.exists(AUDIO_MODEL_PATH):
        audio_regressor = joblib.load(AUDIO_MODEL_PATH)
    else:
        audio_regressor = None

def _load_wav_from_bytes(b: bytes) -> torch.Tensor:
    """
    returns waveform: [1, T] float32
    """
    wav, sr = torchaudio.load(io.BytesIO(b))
    # mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # resample
    if sr != WAV2VEC_BUNDLE.sample_rate:
        wav = torchaudio.transforms.Resample(sr, WAV2VEC_BUNDLE.sample_rate)(wav)
    return wav.to(DEVICE)

@torch.no_grad()
def _embed_wav2vec(wav: torch.Tensor, max_sec: float = 10.0) -> np.ndarray:
    """
    wav: [1,T]
    returns embedding: [768]
    """
    max_len = int(max_sec * WAV2VEC_BUNDLE.sample_rate)
    if wav.size(-1) > max_len:
        wav = wav[..., :max_len]
    feats, _ = wav2vec_model.extract_features(wav)  # list
    last = feats[-1]  # [1,frames,768]
    emb = last.mean(dim=1).squeeze(0)  # [768]
    return emb.detach().cpu().numpy()

def predict_audio_valence(audio_bytes: bytes) -> float:
    """
    WAV bytes -> valence_audio (0..1)
    """
    wav = _load_wav_from_bytes(audio_bytes)
    emb = _embed_wav2vec(wav).reshape(1, -1)
    pred = float(audio_regressor.predict(emb)[0])
    # clip
    pred = max(0.0, min(1.0, pred))
    return pred

# =========================
# Fusion (text ağırlıklı, "devede kulak")
# =========================
def fuse_valence(valence_text: float, valence_audio: float, score: float, mode: str = "dynamic") -> tuple[float, float]:
    """
    returns (valence_fused, w_text)
    mode:
      - "fixed": w_text = 0.92
      - "dynamic": text eminliği düşükse sesi biraz artır (ama yine düşük)
    """
    vt = float(valence_text)
    va = float(valence_audio)

    if mode == "fixed":
        w_text = 0.92
    else:
        # score: seçilen label güveni. Bunu hafifçe kullanacağız.
        # conf = 0..1 (0: kararsız, 1: emin)
        # burada score'u direkt kullanabiliriz ama daha stabil: valence'ın 0.5'e uzaklığı
        conf = min(1.0, max(0.0, abs(vt - 0.5) * 2.0))
        # 0.85..0.95 arası; çoğu zaman 0.90+ olacak.
        w_text = 0.85 + 0.10 * conf

    vf = w_text * vt + (1.0 - w_text) * va
    vf = max(0.0, min(1.0, vf))
    return vf, w_text

# =========================
# Endpoints
# =========================
@app.post("/analyze")
async def analyze_sentiment(request: SentimentRequest):
    """
    Text-only sentiment (geri uyumlu)
    """
    result = sentiment_pipeline(request.text)
    label = result[0]["label"]
    score = float(result[0]["score"])
    return {"sentiment": label, "score": score}

@app.post("/analyze_fusion")
async def analyze_fusion(
    text: str = Form(...),
    audio: UploadFile = File(...),
    fusion_mode: str = Form("dynamic")  # "dynamic" veya "fixed"
):
    """
    Text + Audio fusion.
    - text: string
    - audio: wav file (UploadFile)
    """
    if audio_regressor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Audio valence model yüklenmedi ({AUDIO_MODEL_PATH} bulunamadı). AUDIO_VALENCE_MODEL_PATH ile yol belirtin."
        )
    # 1) text sentiment
    result = sentiment_pipeline(text)
    label = result[0]["label"]
    score = float(result[0]["score"])
    valence_text = text_to_valence(label, score)

    # 2) audio valence
    audio_bytes = await audio.read()
    valence_audio = predict_audio_valence(audio_bytes)

    # 3) fusion (text ağırlıklı)
    valence_fused, w_text = fuse_valence(valence_text, valence_audio, score, mode=fusion_mode)

    return {
        "text": {
            "sentiment": label,
            "score": score,
            "valence_text": valence_text,
        },
        "audio": {
            "valence_audio": valence_audio,
        },
        "fusion": {
            "mode": fusion_mode,
            "w_text": w_text,
            "valence_fused": valence_fused,
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Turkish Sentiment Analysis API. Use /analyze for text-only, /analyze_fusion for text+audio fusion."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

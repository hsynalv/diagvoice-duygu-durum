import io
import os
import tempfile
import traceback

import fastapi
import librosa
import numpy as np
import uvicorn
from fastapi import File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import BaseEstimator
import joblib


TARGET_SR = 16000
MAX_SEC = float(os.environ.get("MENTAL_FITNESS_MAX_SEC", "30"))
MAX_SEC = max(1.0, min(300.0, MAX_SEC))
MAX_SAMPLES = int(TARGET_SR * MAX_SEC)

app = fastapi.FastAPI()

_cors_origins_raw = os.environ.get(
    "CORS_ALLOW_ORIGINS",
    "https://diagvoice.huseyinalav.me,http://localhost:5173,http://127.0.0.1:5173",
)
_cors_allow_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get(
    "MENTAL_FITNESS_MODEL_PATH",
    "/app/models/extratreesclassifier.joblib",
)

print("Starting mental-fitness service and loading model...")
try:
    _model: BaseEstimator | None = joblib.load(MODEL_PATH)
    print(f"Mental fitness model loaded from {MODEL_PATH}")
except Exception:
    print("!!! FATAL: Failed to load mental fitness model !!!")
    print(traceback.format_exc())
    _model = None


def _decode_to_float32_16k(audio_bytes: bytes, filename: str):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        audio_segment = AudioSegment.from_file(tmp_path)
        audio_segment = audio_segment.set_frame_rate(TARGET_SR).set_channels(1).set_sample_width(2)
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        y, sr = librosa.load(wav_buffer, sr=TARGET_SR, mono=True)
        if y is None or len(y) == 0:
            raise RuntimeError("Empty audio")

        if len(y) > MAX_SAMPLES:
            start = (len(y) - MAX_SAMPLES) // 2
            y = y[start : start + MAX_SAMPLES]

        m = float(abs(y).max() + 1e-6)
        y = (y / m).astype("float32")
        return y, sr
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    # Basit ama deterministik bir feature seti: MFCC + temel istatistikler
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)

        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        feats = np.concatenate(
            [
                mfcc_mean,
                mfcc_std,
                np.array([rms.mean(), rms.std(), zcr.mean(), zcr.std()], dtype=np.float32),
            ]
        ).astype(np.float32)
        return feats.reshape(1, -1)
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")


@app.get("/")
async def root():
    return {
        "message": "Mental fitness (canlıda olan) servisi. Ses dosyanızı /analyze-mental endpoint'ine gönderin.",
        "model_path": MODEL_PATH,
    }


@app.post("/analyze-mental")
async def analyze_mental(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=500, detail="Mental fitness model could not be loaded. Check server logs.")

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file.")

        y, sr = _decode_to_float32_16k(audio_bytes, file.filename or "audio")
        feats = _extract_features(y, sr)

        pred_id = int(_model.predict(feats)[0])

        probs = None
        if hasattr(_model, "predict_proba"):
            probs_arr = _model.predict_proba(feats)[0]
            probs = [float(x) for x in probs_arr.tolist()]

        label = None
        if hasattr(_model, "classes_"):
            classes = list(getattr(_model, "classes_"))
            if 0 <= pred_id < len(classes):
                label = str(classes[pred_id])

        return {
            "pred_id": pred_id,
            "pred_label": label,
            "probs": probs,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mental fitness analysis failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)


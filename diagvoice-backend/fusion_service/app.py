import os
import math
import tempfile
from typing import Any, Dict, Optional

import fastapi
import httpx
import uvicorn
from fastapi import File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = fastapi.FastAPI()

_cors_origins_raw = os.environ.get(
    "CORS_ALLOW_ORIGINS",
    "https://diagvoice.huseyinalav.me,http://localhost:5173,http://127.0.0.1:5173,https://duygudurum.diagvoice.com,https://algoritmalar.diagvoice.com,https://yolarkadasim.diagvoice.com",
)
_cors_allow_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VOICE_TO_TEXT_URL = os.environ.get("VOICE_TO_TEXT_URL", "http://localhost:8001/transcribe")
TEXT_TO_SENTIMENT_URL = os.environ.get("TEXT_TO_SENTIMENT_URL", "http://localhost:8000/analyze")
VOICE_TO_SENTIMENT_URL = os.environ.get("VOICE_TO_SENTIMENT_URL", "http://localhost:8002/analyze-audio")
DISEASE_SERVICE_URL = os.environ.get("DISEASE_SERVICE_URL", "http://localhost:8004/analyze-disease")
MENTAL_FITNESS_URL = os.environ.get("MENTAL_FITNESS_URL", "http://localhost:8005/analyze-mental")
AGE_GENDER_URL = os.environ.get("AGE_GENDER_URL", "http://localhost:8006/analyze-age-gender")
DEPRESSION_SERVICE_URL = os.environ.get("DEPRESSION_SERVICE_URL", "http://localhost:8007/analyze-depression")

# benchmark_v2 (inference_api Flask, iç ağ) — boş string ile kapatılabilir
_inference_url_raw = os.environ.get("INFERENCE_API_URL")
if _inference_url_raw is None:
    INFERENCE_API_URL = "http://127.0.0.1:5008"
elif not str(_inference_url_raw).strip():
    INFERENCE_API_URL = ""
else:
    INFERENCE_API_URL = str(_inference_url_raw).strip().rstrip("/")

HTTPX_TOTAL_TIMEOUT_SEC = float(os.environ.get("FUSION_HTTP_TIMEOUT_SEC", "900"))
HTTPX_CONNECT_TIMEOUT_SEC = float(os.environ.get("FUSION_HTTP_CONNECT_TIMEOUT_SEC", "30"))
HTTPX_READ_TIMEOUT_SEC = float(os.environ.get("FUSION_HTTP_READ_TIMEOUT_SEC", "900"))

FUSION_W_TEXT = float(os.environ.get("FUSION_W_TEXT", "0.5"))
FUSION_W_TEXT = max(0.0, min(1.0, FUSION_W_TEXT))

DYNAMIC_FUSION = os.environ.get("DYNAMIC_FUSION", "1").strip() not in {"0", "false", "False"}

# Clamp range for dynamic w_text
W_TEXT_MIN = float(os.environ.get("W_TEXT_MIN", "0.2"))
W_TEXT_MAX = float(os.environ.get("W_TEXT_MAX", "0.8"))
W_TEXT_MIN = max(0.0, min(1.0, W_TEXT_MIN))
W_TEXT_MAX = max(0.0, min(1.0, W_TEXT_MAX))
if W_TEXT_MIN > W_TEXT_MAX:
    W_TEXT_MIN, W_TEXT_MAX = W_TEXT_MAX, W_TEXT_MIN

# Audio emotion class order based on your training script:
# 0 sadness, 1 fear, 2 happiness, 3 anger
#
# Convert to valence (0..1) using a weighted sum of probabilities:
#   v_raw = sum(p_i * w_i) where w_i in [-1..1]
#   valence = (v_raw + 1) / 2
# and then modulate by intensity (0..1) to reduce impact when the model is unsure:
#   valence = 0.5 + intensity * (v_raw / 2)
DEFAULT_AUDIO_VALENCE_WEIGHTS = [-1.0, -0.6, 1.0, -0.8]


def _load_audio_valence_weights() -> list[float]:
    raw = os.environ.get("AUDIO_VALENCE_WEIGHTS", "")
    if not raw.strip():
        return DEFAULT_AUDIO_VALENCE_WEIGHTS
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 4:
        return DEFAULT_AUDIO_VALENCE_WEIGHTS
    try:
        ws = [float(p) for p in parts]
        if any(w < -1.0 or w > 1.0 for w in ws):
            return DEFAULT_AUDIO_VALENCE_WEIGHTS
        return ws
    except Exception:
        return DEFAULT_AUDIO_VALENCE_WEIGHTS


AUDIO_VALENCE_WEIGHTS = _load_audio_valence_weights()


def text_sentiment_to_valence(sentiment_label: str, score: float) -> float:
    lab = (sentiment_label or "").upper()
    s = float(score)
    if lab == "POSITIVE":
        v = s
    elif lab == "NEGATIVE":
        v = 1.0 - s
    else:
        # Unknown label; fall back to neutral-ish
        v = 0.5
    return max(0.0, min(1.0, v))


def text_valence_confidence(valence: Optional[float]) -> Optional[float]:
    if valence is None:
        return None
    # distance to 0.5 mapped to [0..1]
    v = float(valence)
    conf = min(1.0, max(0.0, abs(v - 0.5) * 2.0))
    return conf


def audio_entropy_confidence(probs: Any) -> Optional[float]:
    if not isinstance(probs, list) or len(probs) < 4:
        return None
    try:
        p = [max(1e-12, float(x)) for x in probs[:4]]
        s = sum(p)
        p = [x / s for x in p]
        ent = -sum(x * math.log(x) for x in p)
        ent_max = math.log(4.0)
        ent_norm = ent / ent_max
        # confidence is inverse entropy
        conf = 1.0 - ent_norm
        return max(0.0, min(1.0, conf))
    except Exception:
        return None


def audio_overall_confidence(intensity: Optional[float], probs: Any) -> Optional[float]:
    ent_conf = audio_entropy_confidence(probs)
    if intensity is None and ent_conf is None:
        return None
    inten = None
    if intensity is not None:
        inten = max(0.0, min(1.0, float(intensity)))

    if inten is None:
        return ent_conf
    if ent_conf is None:
        return inten

    # intensity says how strong emotion is; entropy says how peaked distribution is
    return max(0.0, min(1.0, 0.5 * inten + 0.5 * ent_conf))


def audio_probs_to_valence(probs: Any, intensity: Optional[float] = None) -> Optional[float]:
    if not isinstance(probs, list) or len(probs) < 4:
        return None

    try:
        p = [float(x) for x in probs[:4]]
        v_raw = sum(pi * wi for pi, wi in zip(p, AUDIO_VALENCE_WEIGHTS))
        v_raw = max(-1.0, min(1.0, v_raw))

        if intensity is None:
            v = (v_raw + 1.0) / 2.0
        else:
            inten = float(intensity)
            inten = max(0.0, min(1.0, inten))
            v = 0.5 + inten * (v_raw / 2.0)

        return max(0.0, min(1.0, float(v)))
    except Exception:
        return None


def fuse(valence_text: Optional[float], valence_audio: Optional[float], w_text: float) -> Dict[str, Any]:
    if valence_text is None and valence_audio is None:
        return {"valence": None, "label": None, "w_text": w_text}

    if valence_text is None:
        v = valence_audio
        w = 0.0
    elif valence_audio is None:
        v = valence_text
        w = 1.0
    else:
        w = w_text
        v = w * valence_text + (1.0 - w) * valence_audio

    label = "POSITIVE" if v is not None and v >= 0.5 else "NEGATIVE"
    return {"valence": v, "label": label, "w_text": w}


@app.get("/")
async def root():
    return {
        "message": "Fusion service. POST audio to /analyze-fused.",
        "deps": {
            "voice_to_text": VOICE_TO_TEXT_URL,
            "text_to_sentiment": TEXT_TO_SENTIMENT_URL,
            "voice_to_sentiment": VOICE_TO_SENTIMENT_URL,
            "disease_service": DISEASE_SERVICE_URL,
            "mental_fitness": MENTAL_FITNESS_URL,
            "age_gender": AGE_GENDER_URL,
            "depression_service": DEPRESSION_SERVICE_URL,
            "benchmark_inference": INFERENCE_API_URL or None,
        },
        "proxy": {
            "depression": "POST /analyze-depression (multipart file) -> forwards to depression_service",
            "benchmark_v2": (
                f"POST {INFERENCE_API_URL}/predict (internal) — field 'audio', form 'transcript'"
                if INFERENCE_API_URL
                else "disabled (INFERENCE_API_URL empty)"
            ),
        },
        "fusion": {
            "w_text": FUSION_W_TEXT,
            "dynamic_fusion": DYNAMIC_FUSION,
            "w_text_min": W_TEXT_MIN,
            "w_text_max": W_TEXT_MAX,
            "audio_valence_weights": AUDIO_VALENCE_WEIGHTS,
        },
    }


@app.post("/analyze-depression")
async def proxy_analyze_depression(file: UploadFile = File(...)):
    """
    Aracı endpoint: gelen ses dosyasını doğrudan depresyon servisine iletir.
    Yanıt gövdesi ve HTTP durumu depresyon servisiyle aynıdır (JSON).
    """
    tmp_path: Optional[str] = None
    try:
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

        timeout = httpx.Timeout(
            HTTPX_TOTAL_TIMEOUT_SEC,
            connect=HTTPX_CONNECT_TIMEOUT_SEC,
            read=HTTPX_READ_TIMEOUT_SEC,
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            with open(tmp_path, "rb") as f:
                files = {"file": (file.filename or "audio", f, file.content_type or "application/octet-stream")}
                r = await client.post(DEPRESSION_SERVICE_URL, files=files)

        try:
            body = r.json()
        except Exception:
            body = {"detail": r.text or "Upstream returned non-JSON"}

        return JSONResponse(status_code=r.status_code, content=body)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/analyze-fused")
async def analyze_fused(file: UploadFile = File(...)):
    tmp_path: Optional[str] = None
    try:
        # Stream upload to disk to avoid holding large files in memory.
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

        timeout = httpx.Timeout(
            HTTPX_TOTAL_TIMEOUT_SEC,
            connect=HTTPX_CONNECT_TIMEOUT_SEC,
            read=HTTPX_READ_TIMEOUT_SEC,
        )

        benchmark_v2_result: Optional[Dict[str, Any]] = None
        benchmark_v2_err: Optional[str] = None
        stt_text: Optional[str] = None
        stt_err: Optional[str] = "skipped: disabled for fast mode"
        audio_result: Optional[Dict[str, Any]] = None
        audio_err: Optional[str] = "skipped: disabled for fast mode"
        text_result: Optional[Dict[str, Any]] = None
        text_err: Optional[str] = "skipped: disabled for fast mode"
        disease_result: Optional[Dict[str, Any]] = None
        disease_err: Optional[str] = "skipped: disabled for fast mode"
        mental_fitness_result: Optional[Dict[str, Any]] = None
        mental_fitness_err: Optional[str] = "skipped: disabled for fast mode"
        age_gender_result: Optional[Dict[str, Any]] = None
        age_gender_err: Optional[str] = "skipped: disabled for fast mode"
        depression_result: Optional[Dict[str, Any]] = None
        depression_err: Optional[str] = None

        async with httpx.AsyncClient(timeout=timeout) as client:
            # Fast mode: sadece depression + benchmark_v2 çağrılır.
            try:
                with open(tmp_path, "rb") as f:
                    files = {"file": (file.filename or "audio", f, file.content_type or "application/octet-stream")}
                    r = await client.post(DEPRESSION_SERVICE_URL, files=files)
                r.raise_for_status()
                depression_result = r.json()
            except Exception as e:
                depression_err = f"depression_service request failed (url={DEPRESSION_SERVICE_URL}): {e}"

            # benchmark_v2 (is_sağlığı tabular joblib) — iç Flask inference_api
            if INFERENCE_API_URL:
                predict_url = f"{INFERENCE_API_URL}/predict"
                try:
                    with open(tmp_path, "rb") as f:
                        files = {
                            "audio": (
                                (file.filename or "audio.wav").replace("/", "_").replace("\\", "_"),
                                f,
                                file.content_type or "application/octet-stream",
                            )
                        }
                        data = {"transcript": ""}
                        r = await client.post(predict_url, files=files, data=data)
                    r.raise_for_status()
                    benchmark_v2_result = r.json()
                except Exception as e:
                    benchmark_v2_err = f"benchmark_inference request failed (url={predict_url}): {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # Normalize to valence [0,1]
    # text_result: "sentiment" veya "label" (model formatına göre)
    valence_text = None
    if text_result and "score" in text_result:
        sentiment_label = text_result.get("sentiment") or text_result.get("label")
        if sentiment_label is not None:
            valence_text = text_sentiment_to_valence(sentiment_label, text_result.get("score"))

    # FE uyumu: text_sentiment'ta "sentiment" anahtarı olsun (bazı modeller "label" döner)
    text_sentiment_for_response = None
    if text_result:
        text_sentiment_for_response = dict(text_result)
        if "sentiment" not in text_sentiment_for_response and "label" in text_sentiment_for_response:
            text_sentiment_for_response["sentiment"] = text_sentiment_for_response["label"]

    valence_audio = None
    if audio_result and "probs" in audio_result:
        valence_audio = audio_probs_to_valence(audio_result.get("probs"), audio_result.get("intensity"))

    # Dynamic weighting
    text_conf = text_valence_confidence(valence_text)
    audio_conf = audio_overall_confidence(
        audio_result.get("intensity") if isinstance(audio_result, dict) else None,
        audio_result.get("probs") if isinstance(audio_result, dict) else None,
    )

    w_text_used = FUSION_W_TEXT
    if DYNAMIC_FUSION and valence_text is not None and valence_audio is not None and text_conf is not None and audio_conf is not None:
        # Higher confidence -> higher weight.
        # Map relative confidence to [W_TEXT_MIN..W_TEXT_MAX]
        ratio = text_conf / (text_conf + audio_conf + 1e-9)
        w_text_used = W_TEXT_MIN + (W_TEXT_MAX - W_TEXT_MIN) * ratio

    disagreement = None
    if valence_text is not None and valence_audio is not None:
        disagreement = abs(float(valence_text) - float(valence_audio))

    fused = fuse(valence_text, valence_audio, w_text_used)

    return {
        "text": stt_text,
        "text_error": stt_err,
        "audio": audio_result,
        "audio_error": audio_err,
        "disease": disease_result,
        "disease_error": disease_err,
        "mental_fitness": mental_fitness_result,
        "mental_fitness_error": mental_fitness_err,
        "age_gender": age_gender_result,
        "age_gender_error": age_gender_err,
        "depression": depression_result,
        "depression_error": depression_err,
        "benchmark_v2": benchmark_v2_result,
        "benchmark_v2_error": benchmark_v2_err,
        "text_sentiment": text_sentiment_for_response,
        "text_sentiment_error": text_err,
        "valence_text": valence_text,
        "valence_audio": valence_audio,
        "confidence": {
            "text": text_conf,
            "audio": audio_conf,
            "disagreement": disagreement,
            "dynamic_fusion": DYNAMIC_FUSION,
        },
        "fused": fused,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)

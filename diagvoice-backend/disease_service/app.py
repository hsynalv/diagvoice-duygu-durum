import io
import os
import tempfile
import traceback

import fastapi
import librosa
import torch
import uvicorn
from fastapi import File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


TARGET_SR = 16000
MAX_SEC = float(os.environ.get("DISEASE_MAX_SEC", "30"))
MAX_SEC = max(1.0, min(300.0, MAX_SEC))
MAX_SAMPLES = int(TARGET_SR * MAX_SEC)

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

MODEL_DIR = os.environ.get("DISEASE_MODEL_DIR", "/app/models/usye")

print("Starting disease-service and loading model...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model ve feature_extractor'ı yalnızca yerel dosyalardan yükle
    # (HuggingFace Hub'a repo ID olarak gitmesin diye)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    # config.json'da "architectures": ["Wav2Vec2ForSequenceClassification"] olduğu için
    # AutoModelForAudioClassification, Wav2Vec2 tabanlı audio classification modelleriyle uyumlu.
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()} if hasattr(model.config, "id2label") else None
except Exception:
    print("!!! FATAL: Failed to load disease model !!!")
    print(traceback.format_exc())
    model = None
    feature_extractor = None
    id2label = None
    device = "cpu"


def _decode_to_float32_16k(path: str):
    audio_segment = AudioSegment.from_file(path)
    audio_segment = audio_segment.set_frame_rate(TARGET_SR).set_channels(1).set_sample_width(2)
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    y, sr = librosa.load(wav_buffer, sr=TARGET_SR, mono=True)
    if y is None or len(y) == 0:
        raise RuntimeError("Empty audio")
    y = y.astype("float32")

    if len(y) > MAX_SAMPLES:
        start = (len(y) - MAX_SAMPLES) // 2
        y = y[start : start + MAX_SAMPLES]

    m = float(abs(y).max() + 1e-6)
    y = y / m
    return y


@app.get("/")
async def root():
    return {
        "message": "Disease (healthy/sick) audio classifier. Send audio to /analyze-disease (multipart/form-data file).",
        "model_dir": MODEL_DIR,
    }


@app.post("/analyze-disease")
async def analyze_disease(file: UploadFile = File(...)):
    if model is None or feature_extractor is None:
        raise HTTPException(status_code=500, detail="Model could not be loaded. Check server logs.")

    tmp_path = None
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

        y = _decode_to_float32_16k(tmp_path)
        inputs = feature_extractor(y, sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1)

        pred_id_raw = int(torch.argmax(probs).item())
        probs_list = probs.detach().cpu().tolist()

        # Model ters etiket veriyor: 0->healthy, 1->sick aslında ters (sağlıklıya hasta, hastaya sağlıklı basıyor)
        pred_id = 1 - pred_id_raw
        pred_label = id2label.get(pred_id) if isinstance(id2label, dict) else str(pred_id)
        probs_swapped = [probs_list[1], probs_list[0]]

        return {
            "pred_id": pred_id,
            "pred_label": pred_label,
            "probs": probs_swapped,
            "labels": id2label,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease analysis failed: {e}")
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)

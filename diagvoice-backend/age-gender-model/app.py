import io
import os
import json
import tempfile
import traceback

import numpy as np
import fastapi
import torch
import torchaudio
import torch.nn as nn
import uvicorn
from fastapi import File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoFeatureExtractor, AutoModel, AutoModelForAudioClassification

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

MODEL_DIR = os.environ.get(
    "AGE_GENDER_MODEL_DIR",
    "/Users/beyazskorsky/Documents/diag-duygu-durum/diagvoice-backend/models",
)

SR = 16000
MAX_LEN = SR * 4
BASE_MODEL = "microsoft/wavlm-base"
GENDER_MODEL_ID = os.environ.get(
    "GENDER_MODEL_ID",
    "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
)

gender_labels = []
agebin_labels = []
gender2id = {}
agebin2id = {}
model = None
device = None
gender_model = None
gender_feature_extractor = None
gender_model_sr = 16000


class MultiTask(nn.Module):
    def __init__(self, num_gender, num_agebin):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BASE_MODEL)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.g_head = nn.Linear(h, num_gender)
        self.a_head = nn.Linear(h, num_agebin)

    def forward(self, input_values):
        x = self.encoder(input_values=input_values).last_hidden_state.mean(dim=1)
        x = self.drop(x)
        return self.g_head(x), self.a_head(x)


def _load_model():
    global gender_labels, agebin_labels, gender2id, agebin2id, model, device, gender_model, gender_feature_extractor, gender_model_sr

    print("Loading age-gender model...")
    try:
        labels_path = os.path.join(MODEL_DIR, "labels.json")
        with open(labels_path, "r", encoding="utf-8") as f:
            maps = json.load(f)
        gender_labels = maps["gender"]
        agebin_labels = maps["agebin"]
        gender2id = {k: i for i, k in enumerate(gender_labels)}
        agebin2id = {k: i for i, k in enumerate(agebin_labels)}
        print(f"Labels loaded: gender={gender_labels}, agebin={agebin_labels}")

        model_path = os.path.join(MODEL_DIR, "model.pt")
        model = MultiTask(len(gender_labels), len(agebin_labels))
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state, strict=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"Model loaded from {model_path} on {device}")

        try:
            print(f"Loading ready gender model: {GENDER_MODEL_ID}")
            gender_feature_extractor = AutoFeatureExtractor.from_pretrained(GENDER_MODEL_ID)
            gender_model = AutoModelForAudioClassification.from_pretrained(GENDER_MODEL_ID).to(device)
            gender_model.eval()
            gender_model_sr = int(getattr(gender_feature_extractor, "sampling_rate", SR) or SR)
            print(f"Ready gender model loaded (sr={gender_model_sr})")
        except Exception as ge:
            print(f"Ready gender model could not be loaded, fallback will be used: {ge}")
            print(traceback.format_exc())
            gender_model = None
            gender_feature_extractor = None
    except Exception as e:
        print(f"!!! FATAL: Failed to load age-gender model: {e} !!!")
        print(traceback.format_exc())
        model = None
        gender_model = None
        gender_feature_extractor = None


_load_model()


def _preprocess_audio(audio_bytes: bytes) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # pydub ile yükle (m4a, wav, mp3 destekler)
        from pydub import AudioSegment
        audio = AudioSegment.from_file(tmp_path)
        audio = audio.set_frame_rate(SR).set_channels(1)
        wav = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        wav = torch.tensor(wav, dtype=torch.float32)
        wav = torch.nan_to_num(wav)
        wav = wav[:MAX_LEN] if wav.numel() > MAX_LEN else nn.functional.pad(wav, (0, MAX_LEN - wav.numel()))
        return wav
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _normalize_gender_label(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if "fem" in s or "woman" in s or "female" in s or s in {"f", "kadin", "kadın"}:
        return "female"
    if "male" in s or "man" in s or s in {"m", "erkek"}:
        return "male"
    return "female" if s.endswith("0") else "male"


def _predict_gender_ready_model(wav_mono_16k: torch.Tensor):
    if gender_model is None or gender_feature_extractor is None:
        return None

    wav = wav_mono_16k.detach().cpu()
    if wav.dim() != 1:
        wav = wav.squeeze()
    if gender_model_sr != SR:
        wav = torchaudio.functional.resample(wav, orig_freq=SR, new_freq=gender_model_sr)

    x = wav.numpy().astype(np.float32)
    inputs = gender_feature_extractor(
        x,
        sampling_rate=gender_model_sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = gender_model(**inputs)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]

    id2label = getattr(gender_model.config, "id2label", None) or {}
    mapped = []
    for i, p in enumerate(probs.tolist()):
        raw_label = id2label.get(i, str(i))
        mapped.append((_normalize_gender_label(raw_label), float(p)))

    female_prob = 0.0
    male_prob = 0.0
    for lab, p in mapped:
        if lab == "female":
            female_prob += p
        elif lab == "male":
            male_prob += p

    total = female_prob + male_prob
    if total <= 0:
        female_prob, male_prob = 0.5, 0.5
    else:
        female_prob, male_prob = female_prob / total, male_prob / total

    probs_out = [female_prob, male_prob]
    pred_id = int(np.argmax(np.asarray(probs_out)))
    pred_label = ["female", "male"][pred_id]
    return {"pred_id": pred_id, "pred_label": pred_label, "probs": probs_out}


@app.get("/")
async def root():
    return {
        "message": "Age-Gender servisi. Ses dosyanızı /analyze-age-gender endpoint'ine gönderin.",
        "model_dir": MODEL_DIR,
        "gender_labels": gender_labels,
        "agebin_labels": agebin_labels,
    }


@app.post("/analyze-age-gender")
async def analyze_age_gender(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Age-gender model could not be loaded. Check server logs.")

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file.")

        wav = _preprocess_audio(audio_bytes)
        wav = wav.unsqueeze(0).to(device)

        with torch.no_grad():
            lg, la = model(wav)
            g_probs_mt = torch.softmax(lg, dim=-1).cpu().numpy()[0]
            a_probs = torch.softmax(la, dim=-1).cpu().numpy()[0]
            g_pred_mt = int(lg.argmax(-1).cpu().item())
            a_pred = int(la.argmax(-1).cpu().item())

        gender_ready = _predict_gender_ready_model(wav.squeeze(0).cpu())
        if gender_ready is None:
            g_probs = [float(x) for x in g_probs_mt.tolist()]
            g_pred = g_pred_mt
            g_label = gender_labels[g_pred]
            gender_source = "multitask_fallback"
        else:
            g_probs = [float(x) for x in gender_ready["probs"]]
            g_pred = int(gender_ready["pred_id"])
            g_label = str(gender_ready["pred_label"])
            gender_source = "ready_model"

        return {
            "gender": {
                "pred_id": g_pred,
                "pred_label": g_label,
                "probs": g_probs,
                "source": gender_source,
            },
            "agebin": {
                "pred_id": a_pred,
                "pred_label": agebin_labels[a_pred],
                "probs": [float(x) for x in a_probs.tolist()],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Age-gender analysis failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)

import fastapi
import uvicorn
import librosa
import torch
import numpy as np
import io
import tempfile
import os
import traceback
from pathlib import Path
from fastapi import UploadFile, File, HTTPException
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

# FastAPI uygulamasını başlat
app = fastapi.FastAPI()

_cors_origins_raw = os.environ.get(
    "CORS_ALLOW_ORIGINS",
    "https://diagvoice.huseyinalav.me,http://localhost:5173,http://127.0.0.1:5173,https://duygudurum.diagvoice.com,https://algoritmalar.diagvoice.com,https://yolarkadasim.diagvoice.com",
)
_cors_allow_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

# CORS middleware'ini ekle
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Starting server and loading model...")
try:
    model_id = os.environ.get("VOICE_TO_TEXT_MODEL", "openai/whisper-base")
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(model_id)
    speech_to_text_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    speech_to_text_model.to(device)
    speech_to_text_model.eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="turkish", task="transcribe")
    print("Model loaded successfully.")
except Exception as e:
    print("!!! FATAL: Failed to load model !!!")
    print(traceback.format_exc())
    processor = None
    speech_to_text_model = None
    forced_decoder_ids = None

@app.get("/")
async def root():
    return {"message": "Voice-to-Text API. Ses dosyanızı /transcribe endpoint'ine POST isteği ile gönderin."}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Yüklenen ses dosyasını (m4a, mp4, wav, vb.) metne çevirir.
    pydub kullanmak için sistemde ffmpeg yüklü olmalıdır.
    Bu versiyon, dosyayı diskte geçici olarak saklayarak bellek kullanımını optimize eder.
    """
    print("\n--- Received new transcription request ---")
    if speech_to_text_model is None or processor is None:
        print("Error: Model is not loaded.")
        raise HTTPException(status_code=500, detail="Model could not be loaded. Check server logs.")

    temp_file_path = None
    try:
        # Adım 1: Dosyayı geçici bir konuma kaydet
        print(f"Step 1: Saving uploaded file '{file.filename}' to a temporary location.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        print(f"Step 2: File saved to temporary path: {temp_file_path}")

        suffix = Path(file.filename or "").suffix.lower()
        if suffix == ".wav":
            # WAV dosyaları için ffmpeg/pydub gerekmez.
            print("Step 3: Detected WAV input. Loading directly with librosa (no ffmpeg required)...")
            audio_input, sample_rate = librosa.load(temp_file_path, sr=16000, mono=True)
        else:
            # Adım 3: pydub ile sesi yükleyip dönüştür
            # (m4a/mp4 gibi formatlarda ffmpeg/ffprobe gerekir)
            print("Step 3: Converting audio to WAV format using pydub...")
            try:
                audio_segment = AudioSegment.from_file(temp_file_path)
            except FileNotFoundError as e:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Audio conversion failed because ffmpeg/ffprobe was not found on the system. "
                        "Install ffmpeg and ensure it is on PATH, or upload a .wav file. "
                        f"Original error: {e}"
                    ),
                )

            # 16kHz, mono, 16-bit PCM WAV olarak ayarla
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)

            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            print("Step 4: Audio successfully converted to WAV in memory.")

            # Adım 5: librosa ile WAV verisini yükle
            print("Step 5: Loading WAV data with librosa...")
            audio_input, sample_rate = librosa.load(wav_buffer, sr=16000, mono=True)
        print(f"Step 6: Loaded audio with librosa. Sample rate: {sample_rate}, Duration: {len(audio_input)/sample_rate:.2f}s.")

        # Adım 7: Uzun kayıtları 25s'lik parçalara böl ve sırayla çöz
        max_chunk_duration = 25  # saniye
        chunk_size = int(max_chunk_duration * sample_rate)
        total_samples = len(audio_input)
        chunk_texts = []
        print(f"Step 7: Splitting audio into ~{max_chunk_duration}s chunks with {chunk_size} samples each.")

        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk = audio_input[start_idx:end_idx]
            chunk_duration = (end_idx - start_idx) / sample_rate
            print(f"  - Processing chunk {len(chunk_texts)+1} covering samples {start_idx}:{end_idx} (~{chunk_duration:.2f}s)")

            input_features = processor(
                chunk.astype(np.float32),
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            features = input_features["input_features"].to(device=device, dtype=dtype)

            with torch.no_grad():
                generated_ids = speech_to_text_model.generate(
                    features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=256,
                )

            chunk_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if chunk_text:
                chunk_texts.append(chunk_text)

        full_transcription = " ".join(chunk_texts).strip()
        print("Step 8: Chunked transcription successful.")

        return {"text": full_transcription}

    except Exception as e:
        print(f"!!! AN ERROR OCCURRED: {e} !!!")
        print("--- Full Traceback ---")
        print(traceback.format_exc())
        print("----------------------")
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred during transcription. Check server logs. Error: {e}"
        )
    finally:
        # Adım 9: Geçici dosyayı sil
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Step 9: Temporary file {temp_file_path} deleted.")
            except PermissionError as e:
                print(f"Step 9: Failed to delete temporary file {temp_file_path}: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)


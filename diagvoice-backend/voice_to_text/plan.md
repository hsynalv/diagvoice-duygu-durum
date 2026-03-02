# Ses Verisini Metne Çeviren API Geliştirme Planı

Bu döküman, gelen ses verisini metne çevirecek olan yeni API'nin geliştirme adımlarını ve gereksinimlerini açıklamaktadır. Proje, mevcut `text-to-sentiment` API'sinin yapısı ve teknolojileri (FastAPI, Transformers) temel alınarak geliştirilecektir.

## 1. Yeni Bağımlılıklar

Mevcut `text-to-sentiment` projesindeki bağımlılıklara (`torch`, `transformers`, `fastapi`, `uvicorn`) ek olarak aşağıdaki kütüphanelerin yüklenmesi gerekmektedir:

- **`librosa`**: Ses dosyalarını okumak, formatını değiştirmek ve modelin beklediği formata (örneğin 16kHz örnekleme oranına sahip bir NumPy dizisine) dönüştürmek için kullanılacaktır.
- **`python-multipart`**: FastAPI'nin dosya yüklemelerini (`UploadFile`) işleyebilmesi için gereklidir.

## 2. Geliştirme Adımları

### Adım 1: Proje Ortamının Hazırlanması

1.  **Klasör Yapısı**: `voice-to-text` klasörü içinde çalışılacaktır.
2.  Tüm bağımlılıklar yüklendi. 

### Adım 2: API Kodunun Geliştirilmesi (`app.py`)

`voice-to-text` klasörü içine `app.py` adında yeni bir dosya oluşturulacak ve aşağıdaki mantıkla kodlanacaktır:

1.  **Gerekli Kütüphanelerin Import Edilmesi**: `FastAPI`, `UploadFile`, `File`, `pipeline`, `librosa` ve `torch` gibi temel bileşenler import edilecektir.

2.  **Modelin Yüklenmesi**: Hugging Face `transformers` kütüphanesinin `pipeline` fonksiyonu kullanılarak Türkçe ses tanıma modeli yüklenecektir. `openai/whisper-large-v3` modeli yüksek performansı nedeniyle tercih edilebilir.
    ```python
    # Modeli ilk çalıştırmada indirecektir. Bu işlem uzun sürebilir.
    speech_to_text_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
    ```

3.  **FastAPI Endpoint'inin Oluşturulması**:
    -   `/transcribe` adında bir `POST` endpoint'i tanımlanacaktır.
    -   Bu endpoint, `UploadFile` tipinde bir ses dosyası kabul edecektir.

4.  **Ses Verisinin İşlenmesi**:
    -   Yüklenen ses dosyasının byte içeriği okunacaktır (`await file.read()`).
    -   `librosa` kütüphanesi kullanılarak bu byte verisi, modelin işleyebileceği bir formata (16kHz örnekleme oranına sahip float array) dönüştürülecektir.
    -   İşlenen ses verisi, `speech_to_text_pipeline`'a gönderilerek metne çevrilecektir.

5.  **Sonucun Döndürülmesi**: Modelden dönen metin, JSON formatında kullanıcıya sunulacaktır.

### Adım 3: Örnek API Kodu (`app.py`)

Aşağıda, `app.py` dosyası için temel bir başlangıç kodu bulunmaktadır:

```python
import fastapi
import uvicorn
import librosa
import torch
from fastapi import UploadFile, File
from transformers import pipeline

# FastAPI uygulamasını başlat
app = fastapi.FastAPI()

# Cihazı belirle (varsa GPU kullan)
device = 0 if torch.cuda.is_available() else -1

# Ses tanıma modelini yükle
# Bu model ilk çalıştırmada indirilecektir ve boyutu büyüktür.
speech_to_text_pipeline = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-large-v3",
    device=device
)

@app.get("/")
async def root():
    return {"message": "Voice-to-Text API. Ses dosyanızı /transcribe endpoint'ine POST isteği ile gönderin."}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Yüklenen ses dosyasını metne çevirir.
    """
    # Ses dosyasının byte içeriğini oku
    audio_bytes = await file.read()

    # librosa ile sesi yükle ve 16kHz'e yeniden örnekle
    audio_input, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    # Modeli kullanarak metne çevir
    result = speech_to_text_pipeline(audio_input)

    return {"text": result["text"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Adım 4: API'nin Çalıştırılması ve Test Edilmesi

1.  **Sunucuyu Başlatma**: Terminalde, `voice-to-text` dizinindeyken aşağıdaki komut çalıştırılır:
    ```bash
    uvicorn app:app --reload --port 8001
    ```
2.  **Test**: `curl` veya herhangi bir API test aracı ile `http://localhost:8001/transcribe` adresine bir ses dosyası (`.wav`, `.mp3` vb.) `POST` isteği atılarak API test edilebilir.

```bash
# ornek_ses.wav adında bir dosyanız olduğunu varsayalım
curl -X POST -F "file=@ornek_ses.wav" http://localhost:8001/transcribe
```

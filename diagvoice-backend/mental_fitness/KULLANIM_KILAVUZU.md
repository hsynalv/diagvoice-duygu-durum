# Mental Fitness V1 – Kullanım Kılavuzu

Bu kılavuz, Mental Fitness sınıflandırıcısını kendi projenize nasıl dahil edip çalıştıracağınızı adım adım anlatır.

---

## 1. Gereksinimler

### Python sürümü
- Python 3.8 veya üzeri

### Gerekli paketler

```bash
pip install numpy librosa joblib scikit-learn soundfile
```

veya `requirements.txt` ile:

```txt
numpy
librosa
joblib
scikit-learn
soundfile
```

```bash
pip install -r requirements.txt
```

---

## 2. Dosyaları Projenize Kopyalayın

Şu iki dosyayı projenize kopyalayın ve **aynı klasörde** tutun:

```
projeniz/
└── mentalfitness_v1/
    ├── mental_fitness_classifier_v1.py
    └── extratreesclassifier.joblib
```

| Dosya | Açıklama |
|-------|----------|
| `mental_fitness_classifier_v1.py` | Sınıflandırma mantığı ve özellik çıkarma |
| `extratreesclassifier.joblib` | Eğitilmiş model (değiştirmeyin) |

---

## 3. Kullanım

### 3.1 Python'dan çağırma

```python
from mentalfitness_v1.mental_fitness_classifier_v1 import classify_audio

# Ses dosyasını analiz et
result = classify_audio("ses_kaydi.wav")

if "error" in result:
    print("Hata:", result["error"])
else:
    skor = result["results"]["mental_fitness_score"]
    print(f"Mental Fitness Skoru: {skor}")  # 0-100 arası
```

### 3.2 Model yolunu belirtme

Model dosyası farklı bir konumdaysa:

```python
result = classify_audio(
    "ses_kaydi.wav",
    model_path="/proje/yolu/mentalfitness_v1/extratreesclassifier.joblib"
)
```

### 3.3 Komut satırından çalıştırma

```bash
cd projeniz
python -m mentalfitness_v1.mental_fitness_classifier_v1 ses_kaydi.wav
```

veya doğrudan:

```bash
python mentalfitness_v1/mental_fitness_classifier_v1.py ses_kaydi.wav
```

Çıktı örneği:

```json
{
  "results": {
    "mental_fitness_score": 72.45
  }
}
```

---

## 4. Desteklenen Ses Formatları

- WAV
- MP3
- FLAC
- OGG
- M4A

Librosa ile desteklenen diğer formatlar da kullanılabilir.

---

## 5. Çıktı Açıklaması

| Alan | Tip | Açıklama |
|------|-----|----------|
| `mental_fitness_score` | float | 0–100 arası skor. Yüksek = daha sağlıklı, düşük = depresyon eğilimi |
| `error` | string | Hata durumunda mesaj (varsa `results` yoktur) |

---

## 6. Hata Durumları

| Hata mesajı | Neden | Çözüm |
|-------------|-------|-------|
| `Model dosyası bulunamadı` | `extratreesclassifier.joblib` bulunamıyor | Model dosyasının yolunu kontrol edin veya `model_path` verin |
| `Ses dosyası bulunamadı` | Verilen dosya yolu geçersiz | Dosya yolunu ve varlığını kontrol edin |
| `Özellik çıkarma başarısız` | Boş veya bozuk ses dosyası | Geçerli bir ses dosyası kullanın |
| `Sınıflandırma hatası` | Model veya özellik uyumsuzluğu | Model dosyasının orijinal olduğundan emin olun |

---

## 7. Örnek Entegrasyon (Flask / FastAPI)

### Flask

```python
from flask import Flask, request, jsonify
from mentalfitness_v1.mental_fitness_classifier_v1 import classify_audio
import tempfile
import os

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "Dosya gerekli"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi"}), 400
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        result = classify_audio(tmp.name)
        os.unlink(tmp.name)
    
    if "error" in result:
        return jsonify(result), 500
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5000)
```

### FastAPI

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from mentalfitness_v1.mental_fitness_classifier_v1 import classify_audio
import tempfile
import os

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = "." + (file.filename.split(".")[-1] if "." in file.filename else "wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        result = classify_audio(tmp_path)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    finally:
        os.unlink(tmp_path)
```

---

## 8. Performans Notları

- İlk çağrıda model yüklenir (~1–2 saniye).
- Özellik çıkarma ses süresine bağlıdır (genelde 1–5 saniye).
- Model bellekte tutulur; tekrar çağrılarda yükleme tekrarlanmaz (aynı process içinde).

---

## 9. Sık Sorulan Sorular

**S: Ses dosyası ne kadar uzun olmalı?**  
C: En az 1 saniye önerilir. Daha uzun kayıtlar daha kararlı sonuç verebilir.

**S: Model dosyasını değiştirebilir miyim?**  
C: Hayır. `extratreesclassifier.joblib` eğitilmiş modeldir; değiştirirseniz sonuçlar geçersiz olur.

**S: Birden fazla ses dosyasını paralel işleyebilir miyim?**  
C: Evet. Her `classify_audio` çağrısı bağımsızdır. Thread/process havuzu ile paralel çalıştırabilirsiniz.

**S: İnsan sesi kontrolü yapılıyor mu?**  
C: Bu sürümde hayır. API’de kullanılan insan sesi kontrolü ayrı bir modüldür (tensorflow-hub gerektirir).

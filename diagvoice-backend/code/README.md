# `code/features` — benchmark özellikleri

Varsayılan: bu repoda **librosa + basit metin** skalerleri tanımlıdır. Eğitimde kullandığınız
sütun adları farklıysa, benchmark repo’daki `features/` modüllerini buraya kopyalayın veya
`DIAGVOICE_CODE_ROOT` ile o dizini öne alın (`inference_api/engine.py` içinde `setup_sys_path`).

`inference_api/models/model.joblib` dosyasını eğitim çıktınızdan buraya koyun (veya `DIAGVOICE_MODEL`).

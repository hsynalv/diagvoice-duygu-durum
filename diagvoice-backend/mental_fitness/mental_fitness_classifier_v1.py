#!/usr/bin/env python3
"""
Basit Ses Sınıflandırıcı
Tek bir ses dosyasını alır ve depresyon/sağlıklı sınıflandırması yapar.
API yok, sadece düz Python kodu.
"""

import sys
import json
import numpy as np
import librosa
import joblib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


def extract_features(
    path: str,
    sr: int = 22050,
    n_mfcc: int = 40,
    hop_length: int = 512,
    n_fft: int = 2048,
    feature_set: str = "rich",
    trim: bool = False,
    preemph: float = 0.0,
    vad: str = "none",
    vad_top_db: int = 30,
) -> Optional[np.ndarray]:
    """
    Ses dosyasından özellik çıkarma fonksiyonu.
    basic_pipeline_3.py'den alınmıştır.
    """
    try:
        y, sr = librosa.load(path, sr=sr, mono=True)
        if y.size == 0:
            return None

        # Sessizlik kırpma (isteğe bağlı)
        if trim:
            try:
                y, _ = librosa.effects.trim(y, top_db=30)
            except Exception:
                pass

        # Basit VAD (Voice Activity Detection)
        vad = (vad or "none").lower()
        if vad == "librosa":
            try:
                intervals = librosa.effects.split(y, top_db=int(vad_top_db))
                if intervals.shape[0] > 0:
                    y = np.concatenate([y[s:e] for s, e in intervals])
            except Exception:
                pass

        # Normalize etme
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / peak
        y = y - float(np.mean(y))

        # Pre-emphasis (isteğe bağlı)
        if 0.0 < preemph < 1.0 and y.size > 1:
            y = np.append(y[0], y[1:] - preemph * y[:-1])

        # MFCC özellikleri
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        if feature_set == "basic":
            # Basit özellik seti
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
            cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)

            feats = np.hstack([
                mfcc_mean, mfcc_std,
                np.mean(zcr, axis=1),
                np.mean(cent, axis=1),
                np.mean(bw, axis=1),
                np.mean(roll, axis=1),
                np.mean(rms, axis=1)
            ])
            return feats

        # Zengin özellik seti (rich) - orijinal implementasyona uygun
        # MFCC delta özellikleri
        try:
            mfcc_d = librosa.feature.delta(mfcc)
            mfcc_d2 = librosa.feature.delta(mfcc, order=2)
        except Exception:
            mfcc_d = np.zeros_like(mfcc)
            mfcc_d2 = np.zeros_like(mfcc)
        
        mfcc_d_mean = np.mean(mfcc_d, axis=1)
        mfcc_d_std = np.std(mfcc_d, axis=1)
        mfcc_d2_mean = np.mean(mfcc_d2, axis=1)
        mfcc_d2_std = np.std(mfcc_d2, axis=1)

        # Spektral ve enerji özellikleri (ortalama, std)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
        flat = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)

        def ms(x: np.ndarray):
            return float(np.mean(x)), float(np.std(x))

        zcr_m, zcr_s = ms(zcr)
        cent_m, cent_s = ms(cent)
        bw_m, bw_s = ms(bw)
        roll_m, roll_s = ms(roll)
        rms_m, rms_s = ms(rms)
        flat_m, flat_s = ms(flat)

        # Harmonik özellikler
        try:
            y_harm = librosa.effects.harmonic(y)
        except Exception:
            y_harm = y

        # Chroma, spectral contrast, tonnetz
        try:
            chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr, n_fft=n_fft, hop_length=hop_length)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
        except Exception:
            chroma_mean = np.zeros(12, dtype=float)
            chroma_std = np.zeros(12, dtype=float)

        try:
            s_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            s_contrast_mean = np.mean(s_contrast, axis=1)
            s_contrast_std = np.std(s_contrast, axis=1)
        except Exception:
            s_contrast_mean = np.zeros(7, dtype=float)
            s_contrast_std = np.zeros(7, dtype=float)

        try:
            tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
            tonnetz_std = np.std(tonnetz, axis=1)
        except Exception:
            tonnetz_mean = np.zeros(6, dtype=float)
            tonnetz_std = np.zeros(6, dtype=float)

        # Pitch/f0 using pYIN where available; fallback to YIN
        f0_stats = []
        try:
            fmin, fmax = 50.0, 500.0
            try:
                f0, vflag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
            except Exception:
                f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
                vflag = np.isfinite(f0)
            if f0 is not None:
                f0 = np.asarray(f0)
                f0v = f0[np.isfinite(f0)] if np.any(np.isfinite(f0)) else np.array([])
                voiced_frac = float(np.mean(vflag.astype(float))) if isinstance(vflag, np.ndarray) else float(len(f0v) / max(1, len(f0)))
                if f0v.size:
                    f0_stats = [float(np.mean(f0v)), float(np.median(f0v)), float(np.std(f0v)), voiced_frac]
                else:
                    f0_stats = [0.0, 0.0, 0.0, voiced_frac]
            else:
                f0_stats = [0.0, 0.0, 0.0, 0.0]
        except Exception:
            f0_stats = [0.0, 0.0, 0.0, 0.0]

        # Tempo (rough proxy for speech rate) and duration
        try:
            tempo = float(librosa.beat.tempo(y=y, sr=sr, hop_length=hop_length, aggregate=np.median).squeeze())
        except Exception:
            tempo = 0.0
        duration = float(len(y) / sr)

        feats = np.hstack([
            # MFCCs and deltas
            mfcc_mean, mfcc_std,
            mfcc_d_mean, mfcc_d_std,
            mfcc_d2_mean, mfcc_d2_std,
            # Spectral summary
            [zcr_m, zcr_s, cent_m, cent_s, bw_m, bw_s, roll_m, roll_s, rms_m, rms_s, flat_m, flat_s],
            # Harmonic chroma/contrast/tonnetz
            chroma_mean, chroma_std,
            s_contrast_mean, s_contrast_std,
            tonnetz_mean, tonnetz_std,
            # Pitch + prosody
            f0_stats,
            # Global
            [tempo, duration],
        ])
        
        return feats

    except Exception as e:
        print(f"Özellik çıkarma hatası: {e}")
        return None


def classify_audio(audio_path: str, model_path: str = None) -> Dict[str, Any]:
    """
    Ses dosyasını sınıflandırır.
    
    Args:
        audio_path: Ses dosyasının yolu
        model_path: Model dosyasının yolu (varsayılan: extratreesclassifier.joblib)
    
    Returns:
        JSON formatında sonuç dictionary'si
    """
    # Model yolunu belirle
    if model_path is None:
        script_dir = Path(__file__).parent
        model_path = script_dir / "extratreesclassifier.joblib"
    
    model_path = Path(model_path)
    
    # Model dosyasının varlığını kontrol et
    if not model_path.exists():
        return {"error": f"Model dosyası bulunamadı: {model_path}"}
    
    # Ses dosyasının varlığını kontrol et
    audio_path = Path(audio_path)
    if not audio_path.exists():
        return {"error": f"Ses dosyası bulunamadı: {audio_path}"}
    
    try:
        # Modeli yükle
        print("Model yükleniyor...")
        model = joblib.load(model_path)
        print("Model yüklendi.")
        
        # Özellikleri çıkar
        print("Özellikler çıkarılıyor...")
        features = extract_features(str(audio_path), feature_set="rich")
        
        if features is None:
            return {"error": "Özellik çıkarma başarısız (muhtemelen boş ses dosyası)"}
        
        print(f"Çıkarılan özellik sayısı: {len(features)}")
        
        # Özellikleri modele uygun şekle getir
        features_reshaped = features.reshape(1, -1)
        
        # Tahmin yap
        print("Sınıflandırma yapılıyor...")
        probabilities = model.predict_proba(features_reshaped)[0]
        
        # Sağlıklı sınıfın olasılığını bul
        # Model sınıfları: ['depression', 'healthy'] veya ['healthy', 'depression']
        classes = model.classes_
        if 'healthy' in classes:
            healthy_idx = list(classes).index('healthy')
            mental_fitness_score = probabilities[healthy_idx] * 100
        else:
            # Eğer 'healthy' yoksa, 'depression' olmayan sınıfı al
            depression_idx = list(classes).index('depression') if 'depression' in classes else 0
            mental_fitness_score = (1 - probabilities[depression_idx]) * 100
        
        # Virgülden sonra her zaman iki basamak göstermek için string formatı kullanılıyor
        return {
            "results": {
                "mental_fitness_score": float(f"{mental_fitness_score:.2f}")
            }
        }
        
    except Exception as e:
        return {"error": f"Sınıflandırma hatası: {e}"}


def main():
    """Ana fonksiyon - komut satırından kullanım"""
    if len(sys.argv) != 2:
        print("Kullanım: python simple_audio_classifier.py <ses_dosyasi_yolu>")
        print("Örnek: python simple_audio_classifier.py test.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    print(f"Ses dosyası: {audio_file}")
    print("=" * 50)
    
    # Sınıflandırma yap
    result = classify_audio(audio_file)
    
    # Sonuçları yazdır
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()

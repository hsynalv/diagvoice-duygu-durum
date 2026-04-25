"""Free speech ses dosyalarindan guclu akustik/prozodik ozellikler."""

from __future__ import annotations

import math

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore


def _require_librosa() -> None:
    if librosa is None:
        raise ImportError("librosa gerekli: pip install librosa soundfile")


def _safe_stats(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return math.nan, math.nan
    return float(np.mean(x)), float(np.std(x))


def _pause_stats(intervals: np.ndarray, n_samples: int, sr: int) -> dict[str, float]:
    total_dur = float(n_samples / sr) if sr else math.nan
    if intervals.size == 0:
        return {
            "audio_pause_count": 1.0,
            "audio_pause_total_sec": total_dur,
            "audio_pause_mean_sec": total_dur,
            "audio_pause_max_sec": total_dur,
            "audio_speech_active_ratio": 0.0,
            "audio_vad_segment_count": 0.0,
            "audio_vad_segment_mean_sec": math.nan,
            "audio_vad_segment_std_sec": math.nan,
        }

    gaps: list[float] = []
    if intervals[0, 0] > 0:
        gaps.append(float(intervals[0, 0]) / sr)
    for i in range(len(intervals) - 1):
        g = float(intervals[i + 1, 0] - intervals[i, 1]) / sr
        if g > 0:
            gaps.append(g)
    if intervals[-1, 1] < n_samples:
        gaps.append(float(n_samples - intervals[-1, 1]) / sr)

    seg_durs = np.array([float(e - s) / sr for s, e in intervals], dtype=np.float64)
    speech_time = float(np.sum(seg_durs)) if seg_durs.size else 0.0

    pause_total = float(sum(gaps)) if gaps else 0.0
    pause_mean = float(np.mean(np.array(gaps, dtype=np.float64))) if gaps else math.nan
    pause_max = float(max(gaps)) if gaps else 0.0

    return {
        "audio_pause_count": float(len(gaps)),
        "audio_pause_total_sec": pause_total,
        "audio_pause_mean_sec": pause_mean,
        "audio_pause_max_sec": pause_max,
        "audio_speech_active_ratio": (speech_time / total_dur) if total_dur > 0 else math.nan,
        "audio_vad_segment_count": float(len(intervals)),
        "audio_vad_segment_mean_sec": float(np.mean(seg_durs)) if seg_durs.size else math.nan,
        "audio_vad_segment_std_sec": float(np.std(seg_durs)) if seg_durs.size > 1 else 0.0,
    }


def _f0_stats(y: np.ndarray, sr: int, hop_length: int) -> dict[str, float]:
    out = {
        "audio_f0_mean_hz": math.nan,
        "audio_f0_std_hz": math.nan,
        "audio_f0_range_hz": math.nan,
        "audio_f0_voiced_fraction": math.nan,
    }
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            sr=sr,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            hop_length=hop_length,
        )
        voiced = f0[voiced_flag]
        if voiced.size:
            p05, p95 = np.percentile(voiced, [5, 95])
            out.update(
                {
                    "audio_f0_mean_hz": float(np.mean(voiced)),
                    "audio_f0_std_hz": float(np.std(voiced)),
                    "audio_f0_range_hz": float(p95 - p05),
                    "audio_f0_voiced_fraction": float(np.mean(voiced_flag.astype(np.float64))),
                }
            )
            return out
    except Exception:
        pass

    try:
        f0_yin = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            hop_length=hop_length,
        )
        v = f0_yin[np.isfinite(f0_yin)]
        if v.size:
            p05, p95 = np.percentile(v, [5, 95])
            out.update(
                {
                    "audio_f0_mean_hz": float(np.mean(v)),
                    "audio_f0_std_hz": float(np.std(v)),
                    "audio_f0_range_hz": float(p95 - p05),
                    "audio_f0_voiced_fraction": float(v.size / f0_yin.size),
                }
            )
    except Exception:
        pass
    return out


def _praat_voice_quality(y: np.ndarray, sr: int) -> dict[str, float]:
    out = {
        "audio_jitter_local": math.nan,
        "audio_shimmer_local": math.nan,
        "audio_hnr_mean_db": math.nan,
    }
    try:
        import parselmouth
    except ImportError:
        return out
    try:
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        duration = sound.duration
        if duration < 0.5:
            return out
        pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
        pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
        out["audio_jitter_local"] = float(
            parselmouth.praat.call(pulses, "Get jitter (local)", 0.0, duration, 0.0001, 0.02, 1.3)
        )
        out["audio_shimmer_local"] = float(
            parselmouth.praat.call(
                [sound, pulses],
                "Get shimmer (local)",
                0.0,
                duration,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
        )
        harm = sound.to_harmonicity(time_step=0.01, minimum_pitch=75.0)
        out["audio_hnr_mean_db"] = float(parselmouth.praat.call(harm, "Get mean", 0.0, duration))
    except Exception:
        pass
    return out


def extract_audio_features_dict(
    audio_path: str,
    *,
    target_sr: int = 16000,
    top_db: float = 35.0,
    hop_length: int = 512,
    run_praat_voice: bool = False,
) -> dict[str, float]:
    """Ses kaydindan modellemeye uygun akustik/prozodik skalarlar."""
    _require_librosa()
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    dur = float(n / sr) if sr else math.nan

    out: dict[str, float] = {
        "meta_duration_sec": dur,
        "meta_sample_rate_hz": float(sr),
    }

    intervals = librosa.effects.split(y, top_db=top_db, hop_length=hop_length, frame_length=2048)
    out.update(_pause_stats(intervals, n, sr))

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=hop_length)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    flat = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    out["audio_rms_mean"], out["audio_rms_std"] = _safe_stats(rms)
    out["audio_zcr_mean"], out["audio_zcr_std"] = _safe_stats(zcr)
    out["audio_spectral_centroid_mean"], out["audio_spectral_centroid_std"] = _safe_stats(cent)
    out["audio_spectral_bandwidth_mean"], out["audio_spectral_bandwidth_std"] = _safe_stats(bw)
    out["audio_spectral_rolloff_mean"], out["audio_spectral_rolloff_std"] = _safe_stats(roll)
    out["audio_spectral_flatness_mean"], out["audio_spectral_flatness_std"] = _safe_stats(flat)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    for i in range(mfcc.shape[0]):
        m = mfcc[i]
        out[f"audio_mfcc{i+1}_mean"] = float(np.mean(m))
        out[f"audio_mfcc{i+1}_std"] = float(np.std(m))

    out.update(_f0_stats(y, sr, hop_length=hop_length))
    if run_praat_voice:
        out.update(_praat_voice_quality(y, sr))
    else:
        out["audio_jitter_local"] = math.nan
        out["audio_shimmer_local"] = math.nan
        out["audio_hnr_mean_db"] = math.nan

    return out

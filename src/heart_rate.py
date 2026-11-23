"""
Module de calcul du rythme cardiaque à partir d'un signal ECG.
Gère les signaux courts via un fallback SciPy si NeuroKit2 échoue.
"""
import numpy as np
import neurokit2 as nk
from scipy.signal import find_peaks, butter, filtfilt


def _detect_peaks_fallback(signal: np.ndarray, fs: int) -> np.ndarray:
    """Détection R-peaks simple (fallback si NeuroKit2 échoue)."""
    nyq = 0.5 * fs
    try:
        b, a = butter(4, [0.5/nyq, 45.0/nyq], btype='band')
        filtered = filtfilt(b, a, signal)
    except Exception:
        filtered = signal

    height = np.std(filtered) * 0.4
    distance = int(0.3 * fs)  # min 300ms entre 2 battements
    peaks, _ = find_peaks(filtered, height=height, distance=distance)
    return peaks


def calculate_heart_rate(signal: np.ndarray, fs: int = 360) -> dict:
    """Calcule le rythme cardiaque depuis un signal ECG."""
    signal = np.nan_to_num(signal, nan=0.0)

    # Garde-fou : signal trop petit
    if len(signal) < fs * 0.5:
        return {
            "error": f"Signal trop court ({len(signal)} échantillons, min {int(fs*0.5)})",
            "num_beats_detected": 0,
        }

    r_peaks = None
    method = "neurokit2"

    # Tentative NeuroKit2
    try:
        signals, info = nk.ecg_process(signal, sampling_rate=fs)
        r_peaks = info["ECG_R_Peaks"]
        if len(r_peaks) < 2:
            raise ValueError("Pas assez de pics détectés par NeuroKit2")
    except Exception:
        # Fallback SciPy
        method = "fallback"
        try:
            r_peaks = _detect_peaks_fallback(signal, fs)
        except Exception as e:
            return {"error": f"Détection impossible : {str(e)}"}

    if r_peaks is None or len(r_peaks) < 2:
        return {
            "error": "Pas assez de battements détectés",
            "num_beats_detected": int(len(r_peaks)) if r_peaks is not None else 0,
        }

    # Calculs
    rr_intervals = np.diff(r_peaks) / fs
    mean_rr = np.mean(rr_intervals)
    heart_rate_bpm = 60.0 / mean_rr if mean_rr > 0 else 0.0
    sdnn = np.std(rr_intervals) * 1000.0

    if heart_rate_bpm < 60:
        rhythm = "Bradycardie (lent)"
    elif heart_rate_bpm > 100:
        rhythm = "Tachycardie (rapide)"
    else:
        rhythm = "Normal"

    return {
        "heart_rate_bpm": round(float(heart_rate_bpm), 1),
        "rhythm": rhythm,
        "num_beats_detected": int(len(r_peaks)),
        "mean_rr_seconds": round(float(mean_rr), 3),
        "rr_variability_ms": round(float(sdnn), 1),
        "irregular": bool(sdnn > 150),
        "method": method,
    }
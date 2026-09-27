import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

def load_ecg_record(record_path, channel=0):
    """Charge un enregistrement ECG MIT-BIH."""
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, channel]
    annotation = wfdb.rdann(record_path, 'atr')
    return signal, annotation.sample, annotation.symbol

def bandpass_filter(signal, lowcut=0.5, highcut=45.0, fs=360, order=4):
    """Filtre passe-bande pour éliminer bruit et dérive baseline."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

def normalize(signal):
    """Normalisation z-score."""
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

def detect_r_peaks(signal, fs=360):
    """Détecte les pics R avec scipy."""
    peaks, _ = find_peaks(signal, distance=int(0.25 * fs), height=np.std(signal))
    return peaks

def extract_beats(signal, r_peaks, window=180):
    """Extrait une fenêtre autour de chaque pic R."""
    beats = []
    for r in r_peaks:
        if r - window < 0 or r + window > len(signal):
            continue
        beats.append(signal[r-window:r+window])
    return np.array(beats)

def preprocess_pipeline(record_path):
    """Pipeline complet : charge → filtre → normalise → détecte → extrait."""
    signal, r_peaks_ref, labels = load_ecg_record(record_path)
    signal = bandpass_filter(signal)
    signal = normalize(signal)
    r_peaks = detect_r_peaks(signal)
    beats = extract_beats(signal, r_peaks)
    return beats, r_peaks, labels

if __name__ == "__main__":
    # Test rapide sur un enregistrement
    print("🔬 Test du pipeline de prétraitement...")
    beats, peaks, labels = preprocess_pipeline("data/mitbih/100")
    print(f"✅ Nombre de battements détectés : {len(beats)}")
    print(f"✅ Forme d'un battement : {beats[0].shape}")
    print(f"✅ Nombre de pics R : {len(peaks)}")
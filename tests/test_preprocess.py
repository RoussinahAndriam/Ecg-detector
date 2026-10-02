import numpy as np
import pytest
import sys
sys.path.append('src')

try:
    from preprocess import bandpass_filter, normalize, extract_beats
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False


@pytest.mark.skipif(not PREPROCESS_AVAILABLE, reason="preprocess.py absent")
def test_bandpass_filter_preserves_length():
    """Le filtre doit préserver la longueur du signal."""
    signal = np.random.randn(1000)
    filtered = bandpass_filter(signal)
    assert len(filtered) == len(signal)


@pytest.mark.skipif(not PREPROCESS_AVAILABLE, reason="preprocess.py absent")
def test_normalize_mean_zero():
    """Après normalisation, la moyenne doit être ~0."""
    signal = np.array([1, 2, 3, 4, 5])
    normalized = normalize(signal)
    assert abs(np.mean(normalized)) < 1e-6


@pytest.mark.skipif(not PREPROCESS_AVAILABLE, reason="preprocess.py absent")
def test_normalize_std_one():
    """Après normalisation, l'écart-type doit être ~1."""
    signal = np.array([1, 2, 3, 4, 5])
    normalized = normalize(signal)
    assert abs(np.std(normalized) - 1.0) < 1e-6


@pytest.mark.skipif(not PREPROCESS_AVAILABLE, reason="preprocess.py absent")
def test_extract_beats_shape():
    """L'extraction doit renvoyer la bonne forme."""
    signal = np.random.randn(1000)
    peaks = np.array([200, 400, 600])
    beats = extract_beats(signal, peaks, window=100)
    assert beats.shape == (3, 200)


@pytest.mark.skipif(not PREPROCESS_AVAILABLE, reason="preprocess.py absent")
def test_extract_beats_empty():
    """Si aucun pic, l'extraction doit renvoyer un tableau vide."""
    signal = np.random.randn(1000)
    peaks = np.array([]).astype(int)
    beats = extract_beats(signal, peaks, window=100)
    assert len(beats) == 0
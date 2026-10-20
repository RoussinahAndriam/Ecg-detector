"""
Module d'extraction de signal ECG depuis une image.
Utilise OpenCV pour :
1. Charger l'image
2. La binariser (noir/blanc)
3. Supprimer la grille rose/rouge
4. Extraire le tracé du signal
"""
import  cv2
import numpy as np


def load_image(image_bytes: bytes) -> np.ndarray:
    """Charge une image depuis des bytes (upload HTTP)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Impossible de décoder l'image. Format invalide.")
    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Convertit l'image en binaire noir/blanc.
    - Gris
    - Inversion si fond noir
    - Binarisation adaptative
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    if np.mean(gray) < 128:
        gray = 255 - gray

 
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=5,
    )
    return binary


def remove_grid(binary: np.ndarray) -> np.ndarray:
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

 
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
 
    grid = cv2.add(horizontal_lines, vertical_lines)
 
    clean = cv2.subtract(binary, grid)
    return clean


def extract_signal(image: np.ndarray, fs: float = 360) -> np.ndarray:
    
    binary = preprocess_image(image)
    clean = remove_grid(binary)

    h, w = clean.shape
    signal = []

    for x in range(w):
        column = clean[:, x]
        dark_pixels = np.where(column > 0)[0]

        if len(dark_pixels) > 0:
           
            y_position = np.mean(dark_pixels)
        else:
         
            y_position = signal[-1] if signal else h / 2

        signal.append(y_position)

    signal = np.array(signal, dtype=np.float32)

   
    signal = h - signal

     
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    return signal


def image_to_signal(image_bytes: bytes, fs: float = 360) -> np.ndarray:
    """Pipeline complet : bytes → signal numérique."""
    image = load_image(image_bytes)
    return extract_signal(image, fs=fs)
 
import matplotlib  
matplotlib.use("Agg")   

import matplotlib.pyplot as plt  
import sys
import os

sys.path.append('src')
from preprocess import load_ecg_record, bandpass_filter


def generate_ecg_image(
    record_name: str = "100",
    duration_seconds: int = 5,
    output_path: str = "tests/test_ecg.png",
):
    """Génère une image d'ECG de `duration_seconds` secondes."""
    fs = 360
    record_path = f"data/mitbih/{record_name}"

      
    if not os.path.exists(f"{record_path}.dat"):
        raise FileNotFoundError(
            f"Fichier introuvable : {record_path}.dat\n"
            f"Lancez d'abord : python src/download_data.py"
        )

      
    print(f"  Chargement de l'enregistrement {record_name}...")
    signal, _, _ = load_ecg_record(record_path)
    signal = bandpass_filter(signal, fs=fs)

    
    samples = duration_seconds * fs
    signal = signal[:samples]
 
    print(f"  Création de l'image ({duration_seconds}s)...")
    fig, ax = plt.subplots(figsize=(10, 3))

     
    ax.plot(signal, color="black", linewidth=1.2)
    ax.set_title(f"ECG - MIT-BIH {record_name}")
    ax.set_xlabel("Échantillons")
    ax.set_ylabel("Amplitude")
    ax.grid(True, color="pink", linestyle="-", linewidth=0.5)
    ax.set_facecolor("white")

    plt.tight_layout()

     
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"  Image sauvegardée : {output_path}")
    print(f"   Taille : {os.path.getsize(output_path) / 1024:.1f} Ko")
    return output_path


if __name__ == "__main__":
    
    generate_ecg_image("100", duration_seconds=5, output_path="tests/test_ecg.png")
    generate_ecg_image("101", duration_seconds=10, output_path="tests/test_ecg_long.png")
    generate_ecg_image("106", duration_seconds=5, output_path="tests/test_ecg_2.png")

    print("  3 images générées dans tests/")
    print("   - tests/test_ecg.png")
    print("   - tests/test_ecg_long.png")
    print("   - tests/test_ecg_2.png")
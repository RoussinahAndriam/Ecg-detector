import os
import sys
import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Configuration
RANDOM_SEED = 42
N_SAMPLES = 1000
OUTPUT_DIR = "reports"
OUTPUT_HTML = os.path.join(OUTPUT_DIR, "drift_report.html")


def generate_reference_data(n=N_SAMPLES, seed=RANDOM_SEED):
    """
    Simule les données de RÉFÉRENCE (= données d'entraînement).
    
    En production, ce serait les vraies statistiques des signaux ECG
    utilisés pour entraîner le modèle.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "mean": rng.normal(0, 1, n),           # moyenne du signal
        "std": rng.normal(1, 0.2, n),          # écart-type
        "min": rng.normal(-2, 0.3, n),         # minimum
        "max": rng.normal(2, 0.3, n),          # maximum
        "r_peaks_count": rng.integers(70, 90, n),  # nombre de pics R
        "heart_rate": rng.normal(75, 5, n),    # fréquence cardiaque
    })


def generate_current_data(n=N_SAMPLES, drift=False, seed=RANDOM_SEED):
    """
    Simule les données COURANTES (= production).
    
    Args:
        drift: si True, simule une dérive (données différentes de la référence)
    """
    rng = np.random.default_rng(seed + 1)
    if not drift:
        # Pas de dérive : mêmes paramètres que la référence
        return pd.DataFrame({
            "mean": rng.normal(0, 1, n),
            "std": rng.normal(1, 0.2, n),
            "min": rng.normal(-2, 0.3, n),
            "max": rng.normal(2, 0.3, n),
            "r_peaks_count": rng.integers(70, 90, n),
            "heart_rate": rng.normal(75, 5, n),
        })
    else:
        # AVEC dérive : on simule un hôpital qui reçoit d'autres patients
        # (par exemple, plus de personnes âgées avec rythme cardiaque plus lent)
        return pd.DataFrame({
            "mean": rng.normal(0.5, 1.5, n),     # moyenne décalée
            "std": rng.normal(1.5, 0.4, n),      # variabilité augmentée
            "min": rng.normal(-2.5, 0.5, n),
            "max": rng.normal(2.5, 0.5, n),
            "r_peaks_count": rng.integers(50, 75, n),  # moins de pics R
            "heart_rate": rng.normal(60, 10, n),  # rythme plus lent
        })


def run_drift_detection(reference_df, current_df, output_path):
    """
    Lance la détection de dérive et sauvegarde un rapport HTML.
    """
    print("🔍 Analyse de dérive en cours...")
    print(f"   Référence : {len(reference_df)} échantillons")
    print(f"   Courant   : {len(current_df)} échantillons")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.save_html(output_path)
    print(f"✅ Rapport HTML sauvegardé : {output_path}")

    # Analyse du résultat
    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    drift_share = result["metrics"][0]["result"]["drift_share"]

    print()
    print("=" * 60)
    if drift_detected:
        print("⚠️  DÉRIVE DÉTECTÉE !")
        print(f"   Part de colonnes en dérive : {drift_share * 100:.1f}%")
        print("   ➜ Le modèle doit être ré-entraîné ou révisé.")
    else:
        print("✅ Aucune dérive détectée.")
        print(f"   Part de colonnes en dérive : {drift_share * 100:.1f}%")
        print("   ➜ Le modèle reste fiable.")
    print("=" * 60)

    return drift_detected


def main():
    """Point d'entrée principal."""
    print()
    print("🫀 Détecteur de dérive ECG - Evidently AI")
    print("=" * 60)

    # 1. Données de référence
    reference = generate_reference_data()

    # 2. Scénario SANS dérive
    print()
    print("📊 Scénario 1 : Données normales (pas de dérive)")
    current_normal = generate_current_data(drift=False)
    run_drift_detection(reference, current_normal,
                        os.path.join(OUTPUT_DIR, "drift_no_change.html"))

    # 3. Scénario AVEC dérive
    print()
    print("📊 Scénario 2 : Données avec DÉRIVE (nouveau type de patients)")
    current_drift = generate_current_data(drift=True)
    run_drift_detection(reference, current_drift, OUTPUT_HTML)

    print()
    print("🎉 Analyse terminée !")
    print(f"📂 Ouvre le rapport dans ton navigateur : {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
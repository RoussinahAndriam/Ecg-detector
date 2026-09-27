import wfdb
import os

def download_mitbih():
    """Télécharge la base MIT-BIH Arrhythmia (~50 Mo)."""
    os.makedirs("data/mitbih", exist_ok=True)
    
    print("📥 Téléchargement de la base MIT-BIH...")
    print("⏳ Cela peut prendre 5-10 minutes...\n")
    
    wfdb.dl_database('mitdb', 'data/mitbih')
    
    files = os.listdir('data/mitbih')
    print(f"\n✅ {len(files)} fichiers téléchargés dans data/mitbih/")
    
    # Afficher un exemple
    print("\n📋 Exemple de fichiers :")
    for f in sorted(files)[:9]:
        print(f"   - {f}")

if __name__ == "__main__":
    download_mitbih()
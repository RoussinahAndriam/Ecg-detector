"""
API FastAPI pour la détection d'arythmies cardiaques.
Deux endpoints de prédiction :
- POST /predict        : signal numérique (ancien)
- POST /predict/image  : image d'ECG (nouveau)
"""
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import sys
sys.path.append('src')

from preprocess import bandpass_filter, normalize, detect_r_peaks, extract_beats
from image_processor import load_image, extract_signal
from heart_rate import  calculate_heart_rate


app = FastAPI(
    title="ECG Arrhythmia Detector",
    description="Détection d'arythmies sur signal numérique OU image d'ECG",
    version="0.3.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)
try:
    model = joblib.load("models/ecg_model.pkl")
    print("  Modèle chargé avec succès")
except Exception as e:
    model = None
    print(f" Modèle non chargé : {e}")

CLASSES = {0: "Normal", 1: "Ventricular", 2: "Atrial"}


class ECGInput(BaseModel):
    signal: list[float]
    fs: int = 360


  
    @app.get("/", response_class=HTMLResponse)
    def serve_frontend():
     try:
        with open("Interface.html", "r", encoding="utf-8") as f:
            return f.read()
     except FileNotFoundError:
        return HTMLResponse("<h1>interface.html introuvable</h1>", status_code=404)

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

 
@app.post("/predict")
def predict(data: ECGInput):
    """Prédiction sur un signal ECG numérique."""
    if model is None:
        return {"error": "Modèle non chargé. Lancez src/train.py d'abord."}

    signal = np.array(data.signal, dtype=np.float32)

    if len(signal) < 100:
        return {"error": "Signal trop court (minimum 100 échantillons)"}

    signal = bandpass_filter(signal, fs=data.fs)
    signal = normalize(signal)

    r_peaks = detect_r_peaks(signal, fs=data.fs)
    beats = extract_beats(signal, r_peaks)

    if len(beats) == 0:
        return {"error": "Aucun battement détecté"}

    preds = model.predict(beats.reshape(beats.shape[0], -1))
    final = int(np.bincount(preds).argmax())

    return {
        "diagnosis": CLASSES[final],
        "confidence": float(np.mean(preds == final)),
        "beats_analyzed": len(beats),
    }

 
@app.post("/predict/image")
async def predict_from_image(file: UploadFile = File(...)):
    """
    Prédiction à partir d'une IMAGE d'ECG.
    Pipeline : image → signal → BPM → diagnostic
    """
  
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Format accepté : JPEG, PNG"
        )

    
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Fichier vide")

    try:
        image = load_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image invalide : {e}")
 
    try:
        signal = extract_signal(image, fs=360)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur d'extraction du signal : {e}"
        )

    
    heart_info = calculate_heart_rate(signal, fs=360)

   
    diagnosis = None
    if model is not None:
        try:
            filtered = bandpass_filter(signal, fs=360)
            filtered = normalize(filtered)
            r_peaks = detect_r_peaks(filtered, fs=360)
            beats = extract_beats(filtered, r_peaks)

            if len(beats) > 0:
                preds = model.predict(beats.reshape(beats.shape[0], -1))
                final = int(np.bincount(preds).argmax())
                diagnosis = {
                    "class": CLASSES[final],
                    "confidence": float(np.mean(preds == final)),
                    "beats_analyzed": len(beats),
                }
            else:
                diagnosis = {"error": "Pas assez de battements pour classifier"}
        except Exception as e:
            diagnosis = {"error": str(e)}

    return {
        "image_info": {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        },
        "heart_rate": heart_info,
        "diagnosis": diagnosis,
    }
  
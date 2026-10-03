from fastapi import FastAPI
from pydantic import BaseModel

# creation de l'api
app = FastAPI(
    title="Detecteur ECG",
    description="API pour detecter les arythmies cardiaques",
    version="0.1"
)

# structure des données
class ECGInput(BaseModel):
    signal: list[float]
    fs: int = 360


@app.get("/")
def accueil():
    return {
        "message": "API ECG",
        "status": "OK"
    }


@app.get("/health")
def verifier():
    return {"status": "healthy"}


@app.post("/predict")
def predire(data: ECGInput):
    # il faudrait appeler le modele quand ça sera pret
    return {
        "message": "en attente du modele",
        "nb_points": len(data.signal),
        "frequence": data.fs
    }
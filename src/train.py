import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.append('src')
from preprocess import preprocess_pipeline

LABEL_MAP = {'N': 0, 'L': 0, 'R': 0, 'V': 1, 'A': 2, 'F': 2}

def prepare_dataset(records):
    X, y = [], []
    for rec in records:
        print(f"  Traitement de {rec}...")
        try:
            beats, peaks, labels = preprocess_pipeline(f"data/mitbih/{rec}")
            for i, beat in enumerate(beats):
                if i < len(labels) and labels[i] in LABEL_MAP:
                    X.append(beat)
                    y.append(LABEL_MAP[labels[i]])
        except Exception as e:
            print(f"  Erreur sur {rec}: {e}")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    mlflow.set_experiment("ecg-arrhythmia-detection")
    
    records = ["100", "101", "103", "105", "106"]
    X, y = prepare_dataset(records)
    print(f"\n  Dataset: {X.shape}")
    print(f"  Classes: {np.unique(y, return_counts=True)}")
    
    X_flat = X.reshape(X.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.2, random_state=42, stratify=y
    )
    
    with mlflow.start_run():
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        mlflow.log_params(params)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/ecg_model.pkl")
        
        print(f"  Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        print("  Modèle sauvegardé dans models/ecg_model.pkl")
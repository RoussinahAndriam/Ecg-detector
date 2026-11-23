import requests

 
url = "http://localhost:8000/predict/image"

with open("tests/test_ecg.png", "rb") as f:
    files = {"file": ("test_ecg.png", f, "image/png")}
    response = requests.post(url, files=files)

print(f"Status : {response.status_code}")
print(f"Réponse :")
import json
print(json.dumps(response.json(), indent=2))
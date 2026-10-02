import pytest
import sys
sys.path.append('src')

try:
    from fastapi.testclient import TestClient
    from api import app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

if API_AVAILABLE:
    client = TestClient(app)


@pytest.mark.skipif(not API_AVAILABLE, reason="api.py absent")
def test_root_returns_ok():
    """GET / doit renvoyer 200 et status=OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"


@pytest.mark.skipif(not API_AVAILABLE, reason="api.py absent")
def test_health_endpoint():
    """GET /health doit renvoyer healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.skipif(not API_AVAILABLE, reason="api.py absent")
def test_predict_accepts_signal():
    """POST /predict doit accepter un signal ECG."""
    payload = {"signal": [0.1, 0.2, 0.3, 0.4, 0.5], "fs": 360}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["samples_received"] == 5
    assert data["sampling_rate"] == 360


@pytest.mark.skipif(not API_AVAILABLE, reason="api.py absent")
def test_predict_invalid_input():
    """POST /predict sans signal doit renvoyer 422."""
    response = client.post("/predict", json={"fs": 360})
    assert response.status_code == 422
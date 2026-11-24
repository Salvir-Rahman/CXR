"""Integration tests for FastAPI inference backend."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cxr_triage.api.app import create_app
from cxr_triage.config import NIH_CHESTXRAY14_LABELS


@pytest.fixture
def api_client() -> TestClient:
    """Fixture initializing TestClient against the FastAPI application factory."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def valid_png_bytes() -> bytes:
    """Fixture creating valid in-memory PNG radiograph bytes."""
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint(api_client: TestClient) -> None:
    """Verify /health returns operational status, SHA-256 rule hash, and disclaimer."""
    response = api_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "mobilevit" in data["model_backbone"]
    assert data["rules_loaded"] >= 6
    assert len(data["rules_sha256"]) == 64
    assert all(c in "0123456789abcdefABCDEF" for c in data["rules_sha256"])
    assert "Research prototype only" in data["disclaimer"]


def test_predict_valid_image(api_client: TestClient, valid_png_bytes: bytes) -> None:
    """Verify /predict returns complete Deliverable D7 triage payload on valid image."""
    files = {"file": ("radiograph.png", io.BytesIO(valid_png_bytes), "image/png")}
    response = api_client.post("/predict?tau=0.25", files=files)

    assert response.status_code == 200
    data = response.json()

    assert data["decision"] in ("auto", "esc")
    assert 0.0 <= data["study_abnormality_score"] <= 1.0
    assert data["threshold_tau"] == 0.25

    # 14 finding probabilities
    probs = data["finding_probabilities"]
    assert len(probs) == 14
    for label in NIH_CHESTXRAY14_LABELS:
        assert label in probs
        assert 0.0 <= probs[label] <= 1.0

    # 14 discretized severity states
    states = data["discretized_findings"]
    assert len(states) == 14
    for label in NIH_CHESTXRAY14_LABELS:
        assert label in states
        assert states[label] in ("absent", "present", "severe")

    # Auditable rule lists
    assert isinstance(data["supporting_rules"], list)
    assert isinstance(data["violated_rules"], list)
    assert isinstance(data["escalation_reasons"], list)

    # Performance and compliance
    assert data["inference_latency_ms"] > 0.0
    assert "Research prototype only" in data["disclaimer"]


def test_predict_invalid_image_payload(api_client: TestClient) -> None:
    """Verify /predict rejects corrupt or non-image payloads with HTTP 400."""
    corrupt_bytes = b"NOT_A_VALID_IMAGE_BUFFER"
    files = {"file": ("corrupt.png", io.BytesIO(corrupt_bytes), "image/png")}

    response = api_client.post("/predict", files=files)
    assert response.status_code == 400
    assert "not a valid or readable image" in response.json()["detail"]


def test_predict_empty_payload(api_client: TestClient) -> None:
    """Verify /predict rejects empty uploads with HTTP 400."""
    files = {"file": ("empty.png", io.BytesIO(b""), "image/png")}

    response = api_client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Empty image payload" in response.json()["detail"]


def test_predict_custom_tau_escalation(
    api_client: TestClient, valid_png_bytes: bytes
) -> None:
    """Verify setting tau=0.0 forces escalation if risk score > 0."""
    files = {"file": ("radiograph.png", io.BytesIO(valid_png_bytes), "image/png")}
    response = api_client.post("/predict?tau=0.0", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["threshold_tau"] == 0.0
    if data["study_abnormality_score"] > 0.0:
        assert data["decision"] == "esc"
        assert len(data["escalation_reasons"]) > 0


def test_frontend_root_served(api_client: TestClient) -> None:
    """Verify GET / serves frontend index.html with Neuro-Symbolic dashboard."""
    response = api_client.get("/")
    assert response.status_code == 200
    assert "Neuro-Symbolic" in response.text
    assert "Research prototype only" in response.text


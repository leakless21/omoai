import pytest
from litestar import Litestar
from litestar.testing import TestClient
from io import BytesIO

# Import the Litestar app instance
from omoai_api.app.main import app as litestar_app

@pytest.fixture(scope="module")
def test_client() -> TestClient[Litestar]:
    with TestClient(litestar_app) as client:
        yield client

def test_health_endpoint(test_client: TestClient[Litestar]):
    """
    Test the GET /health endpoint.
    """
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_transcribe_endpoint(test_client: TestClient[Litestar]):
    """
    Test the POST /v1/transcribe endpoint.
    """
    # Create a dummy audio file in memory
    audio_content = BytesIO(b"dummy audio data")
    files = {"file": ("test_audio.mp3", audio_content, "audio/mpeg")}

    response = test_client.post("/v1/transcribe", files=files)

    assert response.status_code == 200
    json_response = response.json()
    assert "job_id" in json_response
    assert "status" in json_response
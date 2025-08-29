import pytest
import os
from pathlib import Path
from litestar.testing import TestClient

from src.omoai.api.app import create_app

@pytest.fixture(scope="function")
def client():
    """Create a test client for the Litestar app."""
    app = create_app()
    return TestClient(app)

def test_pipeline_endpoint_with_real_audio(client):
    """
    Tests the /pipeline endpoint with a real audio file.
    This test aims to verify that services_v2.py works correctly.
    """
    audio_file_path = Path("data/input/danba.mp3")

    # Ensure the test audio file exists
    if not audio_file_path.exists():
        pytest.fail(f"Test audio file not found at {audio_file_path}.")

    with open(audio_file_path, "rb") as f:
        files = {"audio_file": ("danba.mp3", f, "audio/mpeg")}
        response = client.post("/pipeline", files=files)

    assert response.status_code in [200, 201], \
        f"Expected status code 200, but got {response.status_code}. Response: {response.text}"

    json_response = response.json()
    assert "summary" in json_response, "Response JSON missing 'summary' key"
    assert "segments" in json_response, "Response JSON missing 'segments' key"

    # Add more specific assertions based on the expected structure of 'summary' and 'segments'
    # For example:
    # assert isinstance(json_response["summary"], dict)
    # assert isinstance(json_response["segments"], list)
    # if json_response["segments"]:
    #     assert "start" in json_response["segments"][0]
    #     assert "end" in json_response["segments"][0]
    #     assert "text" in json_response["segments"][0]

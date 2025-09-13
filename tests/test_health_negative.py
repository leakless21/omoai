import pytest
from litestar.testing import TestClient


def test_health_unhealthy_when_ffmpeg_missing(monkeypatch):
    from omoai.api.app import create_app
    import subprocess
    import omoai.api.health as health

    def fake_run(cmd, *args, **kwargs):
        # Simulate failure only for ffmpeg -version
        if isinstance(cmd, (list, tuple)) and cmd[:2] == ["ffmpeg", "-version"]:
            raise FileNotFoundError("ffmpeg not found")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(health.subprocess, "run", fake_run)

    app = create_app()
    with TestClient(app=app) as client:
        resp = client.get("/health")
        # Unhealthy should map to 500
        assert resp.status_code == 500
        data = resp.json()
        assert data.get("status") == "unhealthy"
        assert data.get("details", {}).get("ffmpeg") == "unavailable"


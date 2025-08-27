"""Test for root endpoint redirect to OpenAPI schema documentation."""
import pytest
from httpx import AsyncClient
from litestar.testing import TestClient

from omoai.api.app import create_app


def test_root_endpoint_redirects_to_schema():
    """Test that the root endpoint (/) redirects to /schema OpenAPI documentation."""
    with TestClient(app=create_app()) as client:
        # Test GET request to root endpoint without following redirects
        response = client.get("/", follow_redirects=False)
        
        # Verify redirect response
        assert response.status_code == 302
        assert "location" in response.headers
        assert response.headers["location"] == "/schema"


def test_schema_endpoint_still_works():
    """Test that the /schema endpoint still returns 200 OK after adding root redirect."""
    with TestClient(app=create_app()) as client:
        # Test GET request to schema endpoint
        response = client.get("/schema")
        
        # Verify successful response
        assert response.status_code == 200


def test_root_redirect_with_follow():
    """Test that following the redirect from / leads to schema documentation."""
    with TestClient(app=create_app()) as client:
        # Test GET request to root endpoint with follow_redirects
        response = client.get("/", follow_redirects=True)
        
        # Verify we end up at schema endpoint with successful response
        assert response.status_code == 200
        # The response should contain OpenAPI schema content
        assert "openapi" in response.text.lower() or "schema" in response.text.lower()
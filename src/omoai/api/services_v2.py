"""Minimal in-memory services v2 stubs for test compatibility.

This module exists so tests can patch `src.omoai.api.services_v2.health_check_models`
without import errors. Real implementations were removed from this branch.
"""

from __future__ import annotations
from typing import Dict


def health_check_models() -> Dict[str, str]:
    """
    Return basic health status for in-memory models.

    Default to 'unhealthy' so script-based fallback remains active unless tests
    patch this to return {'status': 'healthy'}.
    """
    return {"status": "unhealthy"}


__all__ = ["health_check_models"]
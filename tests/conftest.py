"""Shared fixtures for the agent-vitals test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def vitals_snapshot_healthy() -> dict:
    return _load_json(FIXTURES_DIR / "snapshot_healthy.json")

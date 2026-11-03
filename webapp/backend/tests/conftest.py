"""Shared fixtures of the W1 backend tests: one app, one client, the starter requests.

W2's fixtures live in ``tests/_advanced_fixtures.py`` (REVIEW B8), not here.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aodl_webapp.examples import EXAMPLES, example_request
from aodl_webapp.main import create_app
from aodl_webapp.schemas import HoldMove, PlanRequest

# The advanced router is W2's; these tests must run whether or not it imports.
os.environ.setdefault("AODL_STUDIO_STRICT_ADVANCED", "0")

EXAMPLE_IDS: tuple[str, ...] = tuple(example.id for example in EXAMPLES)


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """A ``TestClient`` on one application instance (caches shared across the session)."""
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def pick_and_place() -> PlanRequest:
    return example_request("pick_and_place")


@pytest.fixture
def single_lift_hold() -> PlanRequest:
    return example_request("single_lift_hold")


@pytest.fixture
def array_lift_hold() -> PlanRequest:
    return example_request("array_lift_hold")


@pytest.fixture
def static_tweezer() -> PlanRequest:
    """A single tweezer that never moves: one 100 µs hold at the origin."""
    return PlanRequest(moves=[HoldMove(kind="hold", duration_us=100.0)])


def as_json(req: PlanRequest) -> dict[str, Any]:
    """A request as the JSON body the API receives."""
    return req.model_dump(mode="json")

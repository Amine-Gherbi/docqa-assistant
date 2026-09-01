"""The database engine keeps a small, bounded connection pool."""

from __future__ import annotations

from typing import Any

import pytest

from docqa.config import get_settings
from docqa.db import MAX_OVERFLOW, POOL_RECYCLE_S, POOL_SIZE, make_engine


def test_engine_pool_is_bounded_for_a_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The asyncpg default (5 + 10 overflow) is 15 idle connection buffers."""
    captured: dict[str, Any] = {}

    def fake_create(url: str, **kwargs: Any) -> object:
        captured.update(url=url, **kwargs)
        return object()

    monkeypatch.setattr("docqa.db.create_async_engine", fake_create)

    make_engine(get_settings())

    assert captured["pool_size"] == POOL_SIZE == 2
    assert captured["max_overflow"] == MAX_OVERFLOW == 2
    assert captured["pool_recycle"] == POOL_RECYCLE_S
    assert captured["pool_pre_ping"] is True
